# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

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

logger = logging.getLogger(__name__)


@dataclass
class RoBERTaG2PConfig(FairseqDataclass):
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
class RoBERTaG2PCtcConfig(RoBERTaG2PConfig):
    blank_weight: float = 0
    blank_mode: str = "add"


@register_model("roberta_ctc", dataclass=RoBERTaG2PCtcConfig)
class RoBERTaCtc(BaseFairseqModel):
    def __init__(self, cfg: RoBERTaG2PCtcConfig, roberta_encoder: BaseFairseqModel, target_dictionary: Dictionary):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = roberta_encoder.args.encoder_embed_dim
        self.roberta_encoder = roberta_encoder
        roberta_args = roberta_encoder.args
        self.freeze_finetune_updates = self.cfg.freeze_finetune_updates
        self.target_dictionary = target_dictionary
        self.classification_head = RobertaLMHead(
            embed_dim=roberta_args.encoder_embed_dim,
            output_dim=len(target_dictionary), # 1 for default 4 special tokens of dictionary
            activation_fn=roberta_args.activation_fn,
        )
        self.upsampnet = nn.ConvTranspose1d(
            roberta_args.encoder_embed_dim, roberta_args.encoder_embed_dim,
            kernel_size=cfg.upsamp_kernel_size, stride=cfg.upsamp_stride, padding=cfg.upsamp_padding)
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: RoBERTaG2PCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        # roberta_encoder = RobertaModel.from_pretrained(cfg.roberta_path, checkpoint_file='model.pt')
        models, args, task = checkpoint_utils.load_model_ensemble_and_task([cfg.roberta_path])
        roberta_encoder = models[0]
        roberta_encoder.args = args.model
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
        with torch.no_grad() if not ft else contextlib.ExitStack():
            # import pdb
            # pdb.set_trace()
            x, _ = self.roberta_encoder(
                src_tokens=kwargs['source'],
                features_only=True,
                return_all_hiddens=False,
            )

            # pdb.set_trace()
        x = x.transpose(1,2).contiguous()
        x = self.upsampnet(x)
        x = x.permute(2,0,1).contiguous()
        x = self.classification_head(x)
        return {'encoder_out': x, 'padding_mask': padding_mask}

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

