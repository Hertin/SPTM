# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import peft
import contextlib
import copy
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    register_model,
)
# from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask
from fairseq.data.dictionary import Dictionary
from fairseq.models.roberta import RobertaModel, RobertaLMHead

from transformers import RobertaForMaskedLM

logger = logging.getLogger(__name__)

class WeightedLayerAggregation(nn.Module):
    # take a weighted sum of outputs from all layers
    def __init__(self, n_layer):
        super(WeightedLayerAggregation, self).__init__()
        self.n_layer = n_layer
        self.aggregator = torch.nn.Conv2d(in_channels=n_layer, out_channels=1, kernel_size=1)

    def forward(self, x: List[torch.Tensor]):
        x = torch.stack(x).transpose(0, 1)
        return self.aggregator(x).squeeze(1)


@dataclass
class RoBERTaPEFTConfig(FairseqDataclass):
    roberta_path: str = field(
        default=MISSING, metadata={"help": "path to roberta model"}
    )
    vocab_size: int = field(
        default=100, metadata={"help": "number of classes in the output of the classification head"}
    )
    upsamp_kernel_size: int = field(
        default=10, metadata={"help": "upsample network kernel size"}
    )
    upsamp_stride: int = field(
        default=10, metadata={"help": "upsample network stride"}
    )
    upsamp_padding: int = field(
        default=0, metadata={"help": "upsample network padding"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune roberta for this many updates"}
    )


@dataclass
class RoBERTaPEFTCtcConfig(RoBERTaPEFTConfig):
    blank_weight: float = 0
    blank_mode: str = "add"
    # lora config
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ['v_proj', 'q_proj'])
    lora_dropout: float = 0.1
    bias: str = "none"
    r: int = 8



@register_model("roberta_peft", dataclass=RoBERTaPEFTCtcConfig)
class RoBERTaPEFT(BaseFairseqModel):
    def __init__(self, cfg: RoBERTaPEFTCtcConfig, roberta_encoder: BaseFairseqModel, target_dictionary: Dictionary):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = roberta_encoder.config.hidden_size
        self.hidden_act = roberta_encoder.config.hidden_act
        config = {
            "lora_alpha": cfg.lora_alpha,
            "target_modules": cfg.target_modules,
            "lora_dropout": cfg.lora_dropout,
            "bias": cfg.bias,
            "r": cfg.r,
        }
        peft_config = peft.LoraConfig(**config)
        roberta_encoder_peft = peft.get_peft_model(roberta_encoder, peft_config)
        # import pdb; pdb.set_trace()
        self.roberta_encoder = roberta_encoder_peft

        self.freeze_finetune_updates = self.cfg.freeze_finetune_updates
        self.target_dictionary = target_dictionary
        self.classification_head = RobertaLMHead(
            embed_dim=self.hidden_size,
            output_dim=len(target_dictionary), # 1 for default 4 special tokens of dictionary
            activation_fn=self.hidden_act,
        )
        self.upsampnet = nn.ConvTranspose1d(
            self.hidden_size, self.hidden_size,
            kernel_size=cfg.upsamp_kernel_size, stride=cfg.upsamp_stride, padding=cfg.upsamp_padding)
        self.roberta_layer_aggregator = WeightedLayerAggregation(self.roberta_encoder.config.num_hidden_layers+1) # +1 for original input
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: RoBERTaPEFTCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        # roberta_encoder = RobertaModel.from_pretrained(cfg.roberta_path, checkpoint_file='model.pt')
        # models, args, task = checkpoint_utils.load_model_ensemble_and_task([cfg.roberta_path])
        # roberta_encoder = models[0]
        # roberta_encoder.args = args.model
        roberta_encoder = RobertaForMaskedLM.from_pretrained(cfg.roberta_path)
        return cls(cfg, roberta_encoder, task.target_dictionary)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        padding_mask = torch.repeat_interleave(kwargs['padding_mask'], self.cfg.upsamp_stride, dim=-1)
        ft = self.freeze_finetune_updates <= self.num_updates
        # x, _ = self.roberta_encoder(
        #     src_tokens=kwargs['source'],
        #     features_only=True,
        #     return_all_hiddens=False,
        # )
        input_values = kwargs['source']
        attention_mask = ~padding_mask
        # import pdb; pdb.set_trace()
        result = self.roberta_encoder(input_values, attention_mask, output_hidden_states=True)
        x = self.roberta_layer_aggregator(result.hidden_states) # B x T x C

        # import pdb; pdb.set_trace()

        x = x.transpose(1,2).contiguous() # B x T x C => B x C x T
        x = self.upsampnet(x)
        x = x.permute(2,0,1).contiguous() # B x C x T => T x B x C
        x = self.classification_head(x)
        return {'encoder_out': x, 'padding_mask': padding_mask}

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        # sentence_encoder = self.roberta_encoder.encoder.sentence_encoder
        # test_weight = sentence_encoder.layers[0].self_attn.k_proj.weight
        
        # if (self.num_updates >= self.freeze_finetune_updates) and (not test_weight.requires_grad):
        #     logger.info(f'unfreeze roberta encoder at update: {self.num_updates}')
        #     for n, p in sentence_encoder.layers.named_parameters():
        #         p.requires_grad = True
        # elif (self.num_updates < self.freeze_finetune_updates) and (test_weight.requires_grad): 
        #     logger.info(f'freeze roberta encoder at update: {self.num_updates}')
        #     for n, p in sentence_encoder.layers.named_parameters():
        #         p.requires_grad = False
        # assert sentence_encoder.embed_tokens.weight.requires_grad

