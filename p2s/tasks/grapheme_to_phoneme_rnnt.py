# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import joblib
import logging
import os
import io
import sys
import torch
import copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional
from fairseq.data.dictionary import Dictionary
from omegaconf import MISSING, II, OmegaConf

from fairseq.dataclass import FairseqDataclass

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.data_utils import get_buckets, get_bucketed_sizes
# from fairseq.models.roberta import RobertaModel

from transformers import GPT2Tokenizer
import torch.multiprocessing
from fairseq.optim.amp_optimizer import AMPOptimizer
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

class PLabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )

class GLabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )

# class GLabelEncoder(object):
#     def __init__(self, dictionary, tokenizer):
#         self.dictionary = dictionary
#         self.tokenizer = tokenizer

#     def __call__(self, label):
#         # bpe_sentence = '<s> ' + self.tokenizer.encode(label) + ' </s>'
#         bpe_sentence = ' '.join(['<s>'] + list(map(str, self.tokenizer.encode(label))) + ['</s>'])
#         return self.dictionary.encode_line(
#             bpe_sentence, append_eos=False, add_if_not_exist=False
#         )

class G2PDataset(FairseqDataset):
    def __init__(
        self, grapheme_seq, phoneme_seq, sizes,
        max_sample_size, min_sample_size, blank_id=0, eos_idx=2,
        glabels=None, plabels=None, num_buckets=0, process_glabel=None, process_plabel=None,
        shuffle=True, p2g_ratio=8, gpad=1, ppad=0):
        self.grapheme_seq, self.phoneme_seq = grapheme_seq, phoneme_seq
        self.sizes = sizes
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.shuffle = shuffle
        self.p2g_ratio = p2g_ratio
        self.gpad, self.ppad = gpad, ppad
        self.process_glabel = process_glabel
        self.process_plabel = process_plabel
        self.num_buckets = num_buckets
        self.add_to_input = False
        self.batch_targets = True
        self.blank_id = blank_id
        self.eos_idx = eos_idx

    def __getitem__(self, index):
        graphemes = self.process_glabel(self.grapheme_seq[index])
        phonemes = self.process_plabel(self.phoneme_seq[index])
        return {'id': index, 'source': graphemes, 'label': phonemes}

    def __len__(self):
        return len(self.sizes)

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        target_size = min(max(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = torch.BoolTensor(collated_sources.shape).fill_(False)
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), self.gpad)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}

        input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        out["net_input"] = input

        indices = set(out["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]

        eos = torch.LongTensor([self.eos_idx])
        target = [torch.cat([t, eos], axis=-1) for t in target]

        if self.batch_targets:
            out["target_lengths"] = torch.LongTensor([len(t) for t in target])
            out["net_input"]["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.blank_id, left_pad=False)
            out["net_input"]["target"] = target
            out["ntokens"] = out["target_lengths"].sum().item()

        else:
            out["ntokens"] = sum([len(t) for t in target])
        out["target"] = target
        # import pdb
        # pdb.set_trace()
        return out

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored



@dataclass
class G2PRNNTConfig(FairseqDataclass):
    gdata: str = field(default=MISSING, metadata={"help": "path to data directory"})
    pdata: str = field(default=MISSING, metadata={"help": "path to phoneme data directory"})
    # roberta_path: str = field(
    #     default=MISSING, metadata={"help": "path to roberta model"}
    # )
    glabels: str = field(
        default="ltr",
        metadata={"help": "extension of the grapheme label file to load"},
    )
    plabels: str = field(
        default="km",
        metadata={"help": "extension of the phoneme label file to load"},
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    max_sample_size: int = field(
        default=25000, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: int = field(
        default=0, metadata={"help": "min sample size to skip small examples"}
    )
    gpad_idx: int = field(
        default=1, metadata={"help": "pad index for graphemes"}
    )
    max_position: int = field(
        default=2048, metadata={"help": "pad index for graphemes"}
    )
    # num_phonemes: int = field(
    #     default=100, metadata={"help": "number of phonemes"}
    # )
    p2g_ratio: float = field(
        default=8., metadata={"help": "max phoneme to grapheme ratio"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    source_dictionary_path: str = field(
        default="",
        metadata={"help": "path to source dictionary"},
    )
    target_dictionary_path: str = field(
        default="",
        metadata={"help": "path to target dictionary"},
    )



@register_task("grapheme_to_phoneme_rnnt", dataclass=G2PRNNTConfig)
class G2PRNNTTask(FairseqTask):
    """ """

    cfg: G2PRNNTConfig

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        super().__init__(cfg, **kwargs)

        self.blank_symbol = "<s>"
        self.eos_symbol = "</s>"

        self.tgt_dict = Dictionary.load(cfg.target_dictionary_path)

        self.eos_idx = self.tgt_dict.eos_index

        self.ppad_idx = self.tgt_dict.pad()
        logger.info(f'ppad_idx: {self.ppad_idx}')

        self.src_dict = Dictionary.load(cfg.source_dictionary_path)
        self.src_dict.add_symbol('<mask>')
        self.state.add_factory("target_dictionary", self.load_target_dictionary)
        self.state.add_factory("source_dictionary", self.load_source_dictionary)

        self.blank_idx = (
            self.target_dictionary.index(self.blank_symbol)
            if hasattr(self, "blank_symbol")
            else 0
        )

    def load_source_dictionary(self):
        return self.src_dict

    def load_target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.state.source_dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.state.target_dictionary

    @classmethod
    def setup_task(cls, cfg: G2PRNNTConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (G2PRNNTConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        glabels = os.path.join(self.cfg.gdata, f'{split}.{task_cfg.glabels}')
        plabels = os.path.join(self.cfg.pdata, f'{split}.{task_cfg.plabels}')

        # process_glabel = GLabelEncoder(self.source_dictionary, self.tokenizer)
        process_glabel = GLabelEncoder(self.source_dictionary)
        process_plabel = PLabelEncoder(self.target_dictionary)

        grapheme_seq, phoneme_seq, sizes = [], [], []
        with open(glabels, 'r') as fg, open(plabels, 'r') as fp:
            for i, (lg, lp) in tqdm(enumerate(zip(fg, fp))):
                graphemes = lg.strip()
                phonemes = lp.strip()
                if len(graphemes) == 0:
                    continue
                p2g_ratio = float(len(phonemes)) / len(graphemes)
                if len(graphemes.split()) > self.cfg.max_position:
                    print(f'max grapheme lengths exceed {self.cfg.max_position}')
                    continue
                grapheme_seq.append(graphemes)
                phoneme_seq.append(phonemes)
                sizes.append(len(graphemes.split()) + len(phonemes.split()))
        sizes = np.array(sizes, dtype=np.int64)

        self.datasets[split] = G2PDataset(
            grapheme_seq, phoneme_seq, sizes,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            process_plabel=process_plabel,
            process_glabel=process_glabel,
            num_buckets=self.cfg.num_batch_buckets,
            blank_id=self.blank_idx, eos_idx=self.eos_idx,
            shuffle=True, p2g_ratio=self.cfg.p2g_ratio, gpad=self.cfg.gpad_idx, ppad=self.ppad_idx
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)
        return model

    def build_criterion(self, args: Namespace):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from .rnnt import RNNTCriterion

        return RNNTCriterion(args, self)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output