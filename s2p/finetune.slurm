#!/bin/bash
#SBATCH -J g2pme
#SBATCH -o logs/g2pme_%j.%N.out
#SBATCH -e logs/g2pme_%j.%N.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
#SBATCH --qos=sched_level_2
##SBATCH --exclude=node0013,node0016,node0010,node0020,node0039,node0040

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=g2pu
CONDA_ROOT=/nobackup/users/heting/miniconda3/
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

seed=2024
WORK_DIR=$(pwd)
# FINETUNE_SET=g2p_measure/csj_merged
# FINETUNE_SET=g2p_measure/aishell3
FINETUNE_SET=p2u/csj
# FINETUNE_SET=p2u/aishell3
MANIFEST_DIR=${WORK_DIR}/manifest/${FINETUNE_SET}
# label_type=kana_kakasi_vo3
# label_type=kana_mfa_vo4
# label_type=kana_kakasi
# label_type=kana_randlex
# label_type=text_kana_expand_s
# label_type=langnet
# label_type=text
# label_type=text_g2pw
# label_type=text_mfa
# label_type=ph_g2pu
# label_types=(ph_g2puv1 ph_g2puv2)
# label_types=(ph_g2puv3 ph_g2puv4)
# label_types=(ph_g2puv5 ph_g2puv6)
# label_types=(ph_gt ph_mfa)
# label_types=(ph_g2pw)
# label_types=(ph_gt ph_mfa)
label_types=(ph_kakasi ph_randlex)
pids=()
for label_type in ${label_types[@]}; do
	(
	echo label_type: $label_type

	ckpt_path=/nobackup/users/heting/models/wav2vec_small.pt
	save_dir=${WORK_DIR}/outputs/s2p/wav2vec_small-ft-$(basename ${FINETUNE_SET})-${label_type}-s${seed}

	mkdir -p ${save_dir}
	n_prev=$(ls -1q ${save_dir} | grep hydra_train.log | wc -l)
	echo "${n_prev} previous hydra_train.log"
	cp ${save_dir}/hydra_train.log ${save_dir}/hydra_train.log.${n_prev} || true
	srun --gres=gpu:2 --ntasks=1 -c 16 --mem 256G fairseq-hydra-train \
		checkpoint.save_dir=${save_dir} hydra.run.dir=${save_dir} \
		task.data=${MANIFEST_DIR} task.labels=${label_type} \
		common.fp16=True common.seed=${seed} \
		dataset.valid_subset=dev \
		model.w2v_path=${ckpt_path} \
		--config-dir ${WORK_DIR}/config/finetuning \
		--config-name base_100h
	) &
	pids+=($!)
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false


echo "Run completed at:- "
date
