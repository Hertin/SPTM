#!/bin/bash
#SBATCH -J roberta_rnnt
#SBATCH -o logs/roberta_rnnt_%j.%N.out
#SBATCH -e logs/roberta_rnnt_%j.%N.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclude=node0054,node0050
#SBATCH --exclusive
#SBATCH --qos=sched_level_2


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


seed=2024
WORK_DIR=$(pwd)
# aishell3
DATASET=aishell3
SOURCE_DIR=${WORK_DIR}/manifest/p2u/aishell3
TARGET_DIR=${WORK_DIR}/manifest/p2u/aishell3
# glabel_types=(ph_mfa_ipa ph_gt_ipa)
glabel_types=(ph_g2pu_ipa ph_g2pw_ipa)



# csj
# DATASET=csj
# SOURCE_DIR=${WORK_DIR}/manifest/p2u/csj
# TARGET_DIR=${WORK_DIR}/manifest/p2u/csj
# glabel_types=(ph_mfa_ipa ph_gt_ipa)
# glabel_types=(ph_randlex_ipa ph_kakasi_ipa)
# glabel_types=(ph_g2pu_ipa)

pids=()
for glabel_type in ${glabel_types[@]}; do
	(
	echo glabel_type: $glabel_type
	plabel_type=unit
	glabel_suf=${glabel_type}
	plabel_suf=${plabel_type}
	LAYER=6
	LABEL=${glabel_type}
	SOURCE_DICT_PATH=${WORK_DIR}/manifest/p2u/dict.ipa.txt
	TARGET_DICT_PATH=${TARGET_DIR}/dict.${plabel_type}.txt
	ROBERTA_PATH=${WORK_DIR}/outputs/roberta_librispeech100_ipa_l6/checkpoint_best.pt
	valid_subset=dev
	# FREEZE_FINETUNE_UPDATES=8000
	FREEZE_FINETUNE_UPDATES=10000
	# MAX_TOKEN=32000
	# MAX_TOKENS=16000
	# MAX_TOKENS=5000
	MAX_TOKENS=10000
	# LR=1e-3
	# LR=5e-4
	LR=5e-5
	fp32=True
	UF=2

	SUFFIX="-lr${LR}ffu${FREEZE_FINETUNE_UPDATES}uf${UF}mt${MAX_TOKENS}-s${seed}"
	EXTRA_ARGS=""
	if [ "$fp32" = "True" ]; then
		echo train using fp32
		SUFFIX=${SUFFIX}fp32
		EXTRA_ARGS="${EXTRA_ARGS} common.fp16=False"
	fi

	export WANDB_NAME=roberta_rnnt_${DATASET}_${glabel_type}_${plabel_type}_l${LAYER}-ft${SUFFIX}
	export LD_LIBRARY_PATH=${CONDA_ROOT}/envs/g2pu/lib:${LD_LIBRARY_PATH}

	SAVE_DIR=$(pwd)/outputs/p2su_rnnt/${WANDB_NAME}
	# SAVE_DIR=$(pwd)/outputs/${WANDB_NAME}-ft-ffu10k

	n_prev=$(ls -1q ${SAVE_DIR} | grep hydra_train.log | wc -l)
	echo "${n_prev} previous hydra_train.log"
	cp ${SAVE_DIR}/hydra_train.log ${SAVE_DIR}/hydra_train.log.${n_prev} || true

	# echo """
	srun --gres=gpu:2 --ntasks=1 -c 16 --mem 256G \
		fairseq-hydra-train \
		distributed_training.distributed_world_size=2 \
		common.user_dir=${WORK_DIR} \
		checkpoint.save_dir=${SAVE_DIR} hydra.run.dir=${SAVE_DIR} \
		task.gdata=${SOURCE_DIR} task.pdata=${TARGET_DIR} \
		task.glabels=${glabel_suf} task.plabels=${plabel_suf} \
		task.source_dictionary_path=${SOURCE_DICT_PATH} \
		task.target_dictionary_path=${TARGET_DICT_PATH} \
		optimization.lr="[${LR}]" \
		optimization.update_freq="[$UF]" \
		dataset.valid_subset=$valid_subset \
		dataset.max_tokens=${MAX_TOKENS} \
		model.roberta_path=${ROBERTA_PATH} \
		model.freeze_finetune_updates=${FREEZE_FINETUNE_UPDATES} \
		common.seed=${seed} ${EXTRA_ARGS} \
		--config-dir ${WORK_DIR}/roberta_config/finetuning \
		--config-name p2u_roberta_rnnt
	# """
	) &
	pids+=($!)
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false


echo "Run completed at:- "
date
