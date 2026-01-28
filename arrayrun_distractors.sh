#!/bin/bash

#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g,gpu-a100-80g,gpu-v100-32g
#SBATCH --exclude=gpu47
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --job-name=base_distrs_walk
#SBATCH --error=./outfiles/base_out_%A/%a.err
#SBATCH --output=./outfiles/base_out_%A/%a.out
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


DOMAIN="walker"
TASK="walk"
ACTION_REPEAT=2
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MAX_STEP=1000000
TRANS_TYPE="ensemble"
RESOURCES='distractors/images/*.mp4'

srun python3 train.py \
    --domain_name $DOMAIN \
    --task_name $TASK \
    --action_repeat $ACTION_REPEAT \
    --num_train_steps $MAX_STEP \
    --resource_files $RESOURCES \
    --transition_model_type $TRANS_TYPE \
    --seed $SEED \
    --work_dir ./log/${DOMAIN}_${TASK} \
    --encoder_type pixel \
    --decoder_type identity \
    --img_source video \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --decoder_weight_lambda 1e-7 \
    --hidden_dim 1024 \
    --batch_size 512 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --wandb-sync
