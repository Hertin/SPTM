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
from .modules import Predictor, Joiner
# from fairseq.models.roberta import RobertaModel, RobertaLMHead

import torchaudio
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RoBERTaRNNTConfig(FairseqDataclass):
    hidden_dim: int = field(
        default=768, metadata={"help": "hidden dimension size"}
    )
    transformer_ffn_dim: int = field(
        default=2048, metadata={"help": "transformer ffn dim"}
    )
    transformer_num_layers: int = field(
        default=6, metadata={"help": ""}
    )
    roberta_path: str = field(
        default=MISSING, metadata={"help": "path to roberta model"}
    )
    transformer_num_heads:int = field(
        default=12, metadata={"help": ""}
    )
    vocab_size: int = field(
        default=500, metadata={"help": "number of classes in the output of the classification head"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune roberta for this many updates"}
    )

@dataclass
class RoBERTaRNNTCtcConfig(RoBERTaRNNTConfig):
    blank_weight: float = 0
    blank_mode: str = "add"


class RoBERTaWrapper(torch.nn.Module):
    def __init__(self, roberta_model):
        super().__init__()
        self.roberta_model = roberta_model
    
    def forward(self, input, lengths):
        x, extra = self.roberta_model(input, features_only=True, return_all_hiddens=False,)
        return x, lengths

    def transcribe(self, input, lengths):
        x, extra = self.roberta_model(input, features_only=True, return_all_hiddens=False,)
        return x, lengths

@register_model("roberta_rnnt", dataclass=RoBERTaRNNTConfig)
class RoBERTaRNNT(BaseFairseqModel):
    def __init__(self, 
        cfg: RoBERTaRNNTConfig, 
        roberta_rnnt: BaseFairseqModel, 
        task: FairseqTask,
        target_dictionary: Dictionary,
        source_dictionary: Dictionary,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.roberta_rnnt = roberta_rnnt

        # self.transcriber = roberta_rnnt.transcriber
        # self.predictor = roberta_rnnt.predictor
        # self.joiner = roberta_rnnt.joiner
        
        self.target_dictionary = target_dictionary
        self.source_dictionary = source_dictionary

        self.blank_idx = (
            target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.beamsearch = torchaudio.models.RNNTBeamSearch(self.roberta_rnnt, blank=self.blank_idx)
        self.num_updates = 0
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        sentence_encoder = self.roberta_rnnt.transcriber.roberta_model.encoder.sentence_encoder
        test_weight = sentence_encoder.layers[0].self_attn.k_proj.weight
        
        if (self.num_updates >= self.freeze_finetune_updates) and (not test_weight.requires_grad):
            logger.info(f'unfreeze roberta encoder at update: {self.num_updates}')
            for n, p in sentence_encoder.layers.named_parameters():
                p.requires_grad = True
        elif (self.num_updates < self.freeze_finetune_updates) and (test_weight.requires_grad): 
            logger.info(f'freeze roberta encoder at update: {self.num_updates}')
            for n, p in sentence_encoder.layers.named_parameters():
                p.requires_grad = False
        assert sentence_encoder.embed_tokens.weight.requires_grad


    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: RoBERTaRNNTConfig, task: FairseqTask):
        """Build a new model instance."""
        print('len(task.source_dictionary)', len(task.source_dictionary))
        models, args, roberta_task = checkpoint_utils.load_model_ensemble_and_task([cfg.roberta_path])
        # roberta_encoder = models[0]
        # roberta_encoder.args = args.model

        encoder = RoBERTaWrapper(models[0])

        num_symbols = len(task.target_dictionary)
        encoding_dim = cfg.hidden_dim
        symbol_embedding_dim = cfg.hidden_dim
        num_lstm_layers = 3
        lstm_layer_norm=True
        lstm_layer_norm_epsilon=1e-3
        lstm_dropout=0.3

        predictor = Predictor(
            num_symbols,
            encoding_dim,
            symbol_embedding_dim=symbol_embedding_dim,
            num_lstm_layers=num_lstm_layers,
            lstm_hidden_dim=symbol_embedding_dim,
            lstm_layer_norm=lstm_layer_norm,
            lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
            lstm_dropout=lstm_dropout,
        )

        joiner = Joiner(encoding_dim, num_symbols)

        roberta_rnnt = torchaudio.models.RNNT(encoder, predictor, joiner)
        return cls(cfg, roberta_rnnt, task, task.target_dictionary, task.source_dictionary)

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
        sources = kwargs['source']
        source_lengths = (~kwargs['padding_mask']).sum(-1)
        targets = kwargs['target']
        target_lengths = kwargs['target_lengths']
        predictor_state = kwargs.get('predictor_state')

        targets_pad = torch.cat([
            torch.full((len(targets), 1), self.blank_idx).to(targets), 
            targets, 
        ], dim=-1)

        source_encodings, source_lengths = self.roberta_rnnt.transcriber(
            input=sources,
            lengths=source_lengths,
        )
        target_encodings, target_lengths, predictor_state = self.roberta_rnnt.predictor(
            input=targets_pad,
            lengths=target_lengths,
            state=predictor_state,
        )
        logits, source_lengths, target_lengths = self.roberta_rnnt.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

        return {
            'logits': logits, 
            'sources': sources,
            'source_lengths': source_lengths, 
            'targets': targets,
            'target_lengths': target_lengths, 
            'states': predictor_state
        }
    
    @torch.no_grad()
    def beam_search(self, source, source_length, beam_width):
        # src = torch.cat([torch.zeros(self.segment_length - source_length, *source.shape[1:]).to(source), source[:source_length]])
        # print(1)
        # if len(source)
        src = source[:source_length].unsqueeze(0)
        
        with torch.autocast("cuda"):
            # print(2)
            enc_out, _ = self.roberta_rnnt.transcriber.transcribe(src, source_length)
            # print(3)
            # r = beam_searcher.forward(sources, length=souorce_lengths, beam_width=5)
            hyps = search(self.beamsearch, enc_out, None, beam_width=beam_width)
        # print(4)
        hyp = torch.LongTensor(hyps[0][0])
        # print(5)
        return hyp

from torchaudio.models.rnnt_decoder import (
    _get_hypo_score, 
    _get_hypo_key, 
    _get_hypo_tokens, 
    _get_hypo_predictor_out,
    _get_hypo_state,
    _remove_hypo,
    Hypothesis
)
def _init_b_hypos(self, device: torch.device) -> List[Hypothesis]:
    token = self.blank
    state = None

    one_tensor = torch.tensor([1], device=device)
    pred_out, _, pred_state = self.model.predict(torch.tensor([[token]], device=device), one_tensor, state)
    init_hypo = (
        [token],
        pred_out[0].detach(),
        pred_state,
        0.0,
    )
    return [init_hypo]
    
def _gen_b_hypos(
    self,
    b_hypos: List[Hypothesis],
    a_hypos: List[Hypothesis],
    next_token_probs: torch.Tensor,
    key_to_b_hypo: Dict[str, Hypothesis],
) -> List[Hypothesis]:
    for i in range(len(a_hypos)):
        h_a = a_hypos[i]
        append_blank_score = _get_hypo_score(h_a) + next_token_probs[i, 0] # -1 to 0
        if _get_hypo_key(h_a) in key_to_b_hypo:
            h_b = key_to_b_hypo[_get_hypo_key(h_a)]
            _remove_hypo(h_b, b_hypos)
            score = float(torch.tensor(_get_hypo_score(h_b)).logaddexp(append_blank_score))
        else:
            score = float(append_blank_score)
        h_b = (
            _get_hypo_tokens(h_a),
            _get_hypo_predictor_out(h_a),
            _get_hypo_state(h_a),
            score,
        )
        b_hypos.append(h_b)
        key_to_b_hypo[_get_hypo_key(h_b)] = h_b
    _, sorted_idx = torch.tensor([_get_hypo_score(hypo) for hypo in b_hypos]).sort()
    return [b_hypos[idx] for idx in sorted_idx]

def _gen_a_hypos(
    self,
    a_hypos: List[Hypothesis],
    b_hypos: List[Hypothesis],
    next_token_probs: torch.Tensor,
    t: int,
    beam_width: int,
    device: torch.device,
) -> List[Hypothesis]:
    (
        nonblank_nbest_scores,
        nonblank_nbest_hypo_idx,
        nonblank_nbest_token,
    ) = _compute_updated_scores(a_hypos, next_token_probs, beam_width)

    if len(b_hypos) < beam_width:
        b_nbest_score = -float("inf")
    else:
        b_nbest_score = _get_hypo_score(b_hypos[-beam_width])

    base_hypos: List[Hypothesis] = []
    new_tokens: List[int] = []
    new_scores: List[float] = []
    for i in range(beam_width):
        score = float(nonblank_nbest_scores[i])
        if score > b_nbest_score:
            a_hypo_idx = int(nonblank_nbest_hypo_idx[i])
            base_hypos.append(a_hypos[a_hypo_idx])
            new_tokens.append(int(nonblank_nbest_token[i]))
            new_scores.append(score)

    if base_hypos:
        new_hypos = self._gen_new_hypos(base_hypos, new_tokens, new_scores, t, device)
    else:
        new_hypos: List[Hypothesis] = []

    return new_hypos

def _compute_updated_scores(
    hypos: List[Hypothesis],
    next_token_probs: torch.Tensor,
    beam_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hypo_scores = torch.tensor([_get_hypo_score(h) for h in hypos]).unsqueeze(1)
    nonblank_scores = hypo_scores + next_token_probs[:, 1:]  # [beam_width, num_tokens - 1] # -1 to 0
    nonblank_nbest_scores, nonblank_nbest_idx = nonblank_scores.reshape(-1).topk(beam_width)
    nonblank_nbest_hypo_idx = nonblank_nbest_idx.div(nonblank_scores.shape[1], rounding_mode="trunc")
    nonblank_nbest_token = (nonblank_nbest_idx+1) % nonblank_scores.shape[1]
    return nonblank_nbest_scores, nonblank_nbest_hypo_idx, nonblank_nbest_token

def search(
    self,
    enc_out: torch.Tensor,
    hypo: Optional[List[Hypothesis]],
    beam_width: int,
) -> List[Hypothesis]:
    n_time_steps = enc_out.shape[1]
    device = enc_out.device

    a_hypos: List[Hypothesis] = []
    b_hypos = _init_b_hypos(self, device) if hypo is None else hypo
    
    for t in range(n_time_steps):

        a_hypos = b_hypos
        b_hypos = torch.jit.annotate(List[Hypothesis], [])
        key_to_b_hypo: Dict[str, Hypothesis] = {}
        symbols_current_t = 0

        while a_hypos:
            next_token_probs = self._gen_next_token_probs(enc_out[:, t : t + 1], a_hypos, device)
            next_token_probs = next_token_probs.cpu()
            b_hypos = _gen_b_hypos(self, b_hypos, a_hypos, next_token_probs, key_to_b_hypo)
            if symbols_current_t == self.step_max_tokens:
                break

            a_hypos = _gen_a_hypos(
                self,
                a_hypos,
                b_hypos,
                next_token_probs,
                t,
                beam_width,
                device,
            )
            if a_hypos:
                symbols_current_t += 1

        _, sorted_idx = torch.tensor([self.hypo_sort_key(hyp) for hyp in b_hypos]).topk(beam_width)
        b_hypos = [b_hypos[idx] for idx in sorted_idx]
        
    return b_hypos
