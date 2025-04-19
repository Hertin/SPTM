#!/bin/bash
#SBATCH -J g2p-g2p
#SBATCH -o logs/g2p-g2p_%j.%N.out
#SBATCH -e logs/g2p-g2p_%j.%N.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
#SBATCH --qos=sched_level_2
##SBATCH --exclude=node0013,node0010,node0020,node0039,node0040

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=g2pu
CONDA_ROOT=/nobackup/users/heting/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

ulimit -s unlimited

## Creating SLURM nodes list
# export NODELIST=nodelist.$
# srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

SEED=2024

# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_mfa_ipa

# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2pw_ipa

# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_gt_ipa

DATASET=manifest/p2u/aishell3
P_LABEL=ph_g2pu_ipa

# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2puv1_ipa
# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2puv2_ipa
# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2puv3_ipa
# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2puv4_ipa
# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2puv5_ipa
# DATASET=manifest/p2u/aishell3
# P_LABEL=ph_g2puv6_ipa

# DATASET=manifest/p2u/csj
# P_LABEL=ph_mfa_ipa

# DATASET=manifest/p2u/csj
# P_LABEL=ph_kakasi_ipa

# DATASET=manifest/p2u/csj
# P_LABEL=ph_gt_ipa

# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2pu_ipa

# DATASET=manifest/p2u/csj
# P_LABEL=ph_randlex_ipa # CSJ only

# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2puv1_ipa
# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2puv2_ipa
# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2puv3_ipa
# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2puv4_ipa
# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2puv5_ipa
# DATASET=manifest/p2u/csj
# P_LABEL=ph_g2puv6_ipa

U_LABEL=unit
TEXT=$(pwd)/${DATASET}

# echo """
# fairseq-preprocess --source-lang ${P_LABEL} --target-lang ${U_LABEL} \
#     --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
#     --destdir $(pwd)/data-bin/$(basename ${DATASET})-${P_LABEL}-${U_LABEL} \
#     --workers 20
# """


DATASET_NAME=$(basename ${DATASET})
save_dir=$(pwd)/outputs/p2su_ce/${DATASET_NAME}-${P_LABEL}-${U_LABEL}-s${SEED}

srun --gres=gpu:4 --ntasks=1 fairseq-train \
	data-bin/${DATASET_NAME}-${P_LABEL}-${U_LABEL} \
    --distributed-world-size 4 \
    --save-dir ${save_dir} \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 40, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --no-epoch-checkpoints --seed ${SEED} 

