#!/bin/bash
#SBATCH -J roberta
#SBATCH -o logs/roberta_%j.%N.out
#SBATCH -e logs/roberta_%j.%N.err
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
#SBATCH --exclude=node0063
##SBATCH --qos=sched_level_2

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=g2pu
CONDA_ROOT=/nobackup/users/heting/miniconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

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
# LAYER=6
# FRNAME=phg2pw
# REPEAT=15
# MAX_POS=2048
# # REPEAT=30
# # REPEAT=75
# DATASET=aishellX
# LABEL=${FRNAME}${REPEAT}
# DATA_DIR=$(pwd)/data-bin/${DATASET}_${LABEL}
# PROJ_NAME=roberta
# export WANDB_NAME=roberta_${DATASET}_${LABEL}_l${LAYER}
# SAVE_DIR=$(pwd)/outputs/${WANDB_NAME}


LAYER=6
# FRNAME=ph_g2puv1
# FRNAME=ph_g2puv2
# FRNAME=ph_g2puv3
# FRNAME=ph_g2puv4
# FRNAME=ph_g2puv5
FRNAME=ph_g2puv6
# FRNAME=ipa
# FRNAME=ipa15
# REPEAT=15

MAX_POS=2048
# REPEAT=30
# REPEAT=75
DATASET=aishell3
# DATASET=csj
# DATASET=librispeech100
# LABEL=${FRNAME}_${REPEAT}
LABEL=${FRNAME}
DATA_DIR=$(pwd)/data-bin/${DATASET}_${LABEL}
PROJ_NAME=roberta
export WANDB_NAME=roberta_${DATASET}_${LABEL}_l${LAYER}
SAVE_DIR=$(pwd)/outputs/${WANDB_NAME}

if [[ -f "${SAVE_DIR}/hydra_train.log" ]]; then
    n_prev=$(ls -1q ${SAVE_DIR} | grep hydra_train.log | wc -l)
    echo "${n_prev} previous hydra_train.log"
    cp ${SAVE_DIR}/hydra_train.log ${SAVE_DIR}/hydra_train.log.${n_prev} || true
fi

srun --gres=gpu:4 --ntasks=1 fairseq-hydra-train -m \
	--config-dir roberta_config/pretraining \
	--config-name base \
	common.wandb_project=${WANDB_NAME} \
	task.data=$DATA_DIR +task.shorten_method=random_crop \
	model.max_positions=${MAX_POS} +model.encoder_layers=${LAYER} \
	checkpoint.save_dir=${SAVE_DIR} \
	distributed_training.distributed_world_size=4 \
	dataset.batch_size=16

