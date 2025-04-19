#!/bin/bash

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=g2pu
CONDA_ROOT=/nobackup/users/heting/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

## Activate WMLCE virtual environment
FAIRSEQ_ROOT=/nobackup/users/heting/wavenet-r9y9/fairseq-0.12.1
export FAIRSEQ_ROOT=/nobackup/users/heting/g2pu/fairseq-0.12.1
WORK_DIR=$(pwd)
# MANIFEST_DIR=${WORK_DIR}/manifest/gp_zh
MANIFEST_DIR=${WORK_DIR}/manifest/p2u/csj

# LABEL=chrnp
# MODEL=wav2vec_zh-ft-gp_zh-chrnp
# LABEL=ph_kakasi
LABEL=ph_g2puv3
LABEL=ph_gt
LABEL=ph_mfa
LABEL=ph_randlex
MODEL=wav2vec_small-ft-csj-$LABEL

W2V=${WORK_DIR}/outputs/s2p/${MODEL}-s2024/checkpoint_best.pt
# LM=${WORK_DIR}/manifest/aishell/4-gram-phrase.arpa
# LM=${WORK_DIR}/manifest/gp_zh/4-gram-phrase.arpa

# LEXICON=${WORK_DIR}/manifest/aishell/lexicon_chrnp.lst
# LEXICON=${WORK_DIR}/manifest/gp_zh/lexicon_chrnp.lst
# LEXICON=${WORK_DIR}/manifest/aishell/lexicon_wrd.lst
# LEXICON=${WORK_DIR}/manifest/gp_zh/lexicon_wrd.lst
# SPLIT=train
# SPLIT=dev
# SPLIT=test
# SPLIT=train_sub
SPLIT=dev
DECODE='viterbi'
results_dir=results/asr_${MODEL}_${DECODE}
POST_PROCESS=letter


mkdir -p $results_dir

echo Using viterbi decoding
export LD_LIBRARY_PATH=/nobackup/users/heting/miniconda3/envs/g2pu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$FAIRSEQ_ROOT
python ${FAIRSEQ_ROOT}/examples/speech_recognition/infer.py ${MANIFEST_DIR} --task audio_finetuning \
	--nbest 1 --path ${W2V} --gen-subset ${SPLIT} --results-path ${results_dir} \
	--w2l-decoder viterbi --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ${LABEL} --max-tokens 6000000 \
	--post-process ${POST_PROCESS}
~/software/SCTK/bin/sclite -i wsj -r ${results_dir}/ref.units-checkpoint_best.pt-${SPLIT}.txt  trn -h ${results_dir}/hypo.units-checkpoint_best.pt-${SPLIT}.txt trn | tee ${results_dir}/${SPLIT}.uer
