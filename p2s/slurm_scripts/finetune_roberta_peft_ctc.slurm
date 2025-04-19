#!/bin/bash
#SBATCH -J roberta_ctc
#SBATCH -o logs/roberta_ctc_%j.%N.out
#SBATCH -e logs/roberta_ctc_%j.%N.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --wait-all-nodes=1
##SBATCH --qos=sched_level_2
##SBATCH --exclusive
##SBATCH --exclude=node0013,node0010,node0020,node0039,node0040


PYTHON_VIRTUAL_ENVIRONMENT=g2pu
CONDA_ROOT=/nobackup/users/heting/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

## Creating SLURM nodes list
export NODELIST=nodelist.$
# srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date
seed=2025
REPEAT=15
# glabel_type=ph_g2puv1_ipa
# glabel_type=ph_g2puv2_ipa
# glabel_type=ph_g2puv3_ipa
# glabel_type=ph_g2puv4_ipa
# glabel_type=ph_g2puv5_ipa
glabel_type=ph_g2puv6_ipa
plabel_type=unit
glabel_suf=${glabel_type}${REPEAT}
plabel_suf=$plabel_type
LAYER=6
WORK_DIR=$(pwd)
DATASET=aishell3
LABEL=${glabel_type}${REPEAT}
SOURCE_DIR=${WORK_DIR}/manifest/p2u/aishell3
TARGET_DIR=${WORK_DIR}/manifest/p2u/aishell3
ROBERTA_PATH=${WORK_DIR}/outputs/roberta_librispeech100_ipa15_l6/transformers_version
SOURCE_DICT_PATH=${WORK_DIR}/manifest/p2u/dict.ipa15.txt
TARGET_DICT_PATH=${TARGET_DIR}/dict.${plabel_type}.txt
kernel_size=1
stride=1
padding=0
VOCAB_SIZE=500
valid_subset=dev
# LR=1e-3
# LORA_ALPHA=16
# R=8

# seed=2025
# REPEAT=15
# # glabel_type=ph_g2puv1_ipa
# # glabel_type=ph_g2puv2_ipa
# # glabel_type=ph_g2puv3_ipa
# # glabel_type=ph_g2puv4_ipa
# # glabel_type=ph_g2puv5_ipa
# glabel_type=ph_g2puv6_ipa
# plabel_type=unit
# glabel_suf=${glabel_type}${REPEAT}
# plabel_suf=$plabel_type
# LAYER=6
# WORK_DIR=$(pwd)
# DATASET=csj
# LABEL=${glabel_type}${REPEAT}
# SOURCE_DIR=${WORK_DIR}/manifest/p2u/csj
# TARGET_DIR=${WORK_DIR}/manifest/p2u/csj
# ROBERTA_PATH=${WORK_DIR}/outputs/roberta_librispeech100_ipa15_l6/transformers_version
# SOURCE_DICT_PATH=${WORK_DIR}/manifest/p2u/dict.ipa15.txt
# TARGET_DICT_PATH=${TARGET_DIR}/dict.${plabel_type}.txt
# kernel_size=1
# stride=1
# padding=0
# VOCAB_SIZE=500
# valid_subset=dev

LR=1e-3
LORA_ALPHA=16
R=8

SUFFIX="-lr${LR}alpha${LORA_ALPHA}r${R}-s${seed}"

export WANDB_NAME=roberta_peft_${DATASET}_${glabel_suf}_${plabel_type}_l${LAYER}-ft${SUFFIX}

SAVE_DIR=$(pwd)/outputs/${WANDB_NAME}

n_prev=$(ls -1q ${SAVE_DIR} | grep hydra_train.log | wc -l)
echo "${n_prev} previous hydra_train.log"
cp ${SAVE_DIR}/hydra_train.log ${SAVE_DIR}/hydra_train.log.${n_prev} || true

srun --gres=gpu:2 --ntasks=1 \
	fairseq-hydra-train \
    distributed_training.distributed_world_size=1 \
	common.user_dir=${WORK_DIR} \
    checkpoint.save_dir=${SAVE_DIR} hydra.run.dir=${SAVE_DIR} \
    task.gdata=${SOURCE_DIR} task.pdata=${TARGET_DIR} \
    task.glabels=${glabel_suf} task.plabels=${plabel_suf} \
	task.source_dictionary_path=${SOURCE_DICT_PATH} \
    task.target_dictionary_path=${TARGET_DICT_PATH} \
	optimization.lr="[${LR}]" \
    dataset.valid_subset=$valid_subset \
    model.roberta_path=${ROBERTA_PATH} model.vocab_size=${VOCAB_SIZE} \
	model.r=${R} model.lora_alpha=${LORA_ALPHA} \
    model.upsamp_kernel_size=${kernel_size} model.upsamp_stride=${stride} model.upsamp_padding=${padding} \
	common.seed=${seed} \
    --config-dir ${WORK_DIR}/roberta_config/finetuning \
    --config-name p2u_roberta_peft_ctc

echo "Run completed at:- "
date
