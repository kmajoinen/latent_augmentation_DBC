#!/bin/bash

#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g,gpu-a100-80g,gpu-v100-32g
#SBATCH --exclude=gpu[14,43]
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --job-name=base
#SBATCH --error=./outfiles/base_out_%A/%a.err
#SBATCH --array=0-4

mkdir -p outputs

module load mamba
source activate ../tr_deep_bisim4control/env/

SEEDS=(
  1
  2
  3
  4
  5
)


DOMAIN="cheetah"
TASK="run"
ACTION_REPEAT=4
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MAX_STEP=1500000

srun python3 train.py \
    --domain_name $DOMAIN \
    --task_name $TASK \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat $ACTION_REPEAT \
    --work_dir ./log \
    --seed $SEED \
    --wandb-sync
