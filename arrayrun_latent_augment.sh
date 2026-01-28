#!/bin/bash

#SBATCH --mem=75G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g,gpu-a100-80g,gpu-v100-32g
# #SBATCH --exclude=gpu[45,47,48]
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=aug_K1M1_128
#SBATCH --error=./outfiles/aug_out_%A/%a.err
#SBATCH --output=./outfiles/aug_out_%A/%a.out
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
MAX_STEP=1000000
TRANS_TYPE="deterministic"
JITTER=1e-4
AUG_K=1
AUG_M=1
BATCH_S=128

srun python3 train.py \
    --proj-name "Latent_augment" \
    --domain_name $DOMAIN \
    --task_name $TASK \
    --action_repeat $ACTION_REPEAT \
    --seed $SEED \
    --num_train_steps $MAX_STEP \
    --transition_model_type $TRANS_TYPE \
    --encoder_type pixel \
    --decoder_type identity \
    --work_dir ./log \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --decoder_weight_lambda 1e-7 \
    --hidden_dim 1024 \
    --batch_size $BATCH_S \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --wandb-sync \
    --augment \
    --jitter-strength $JITTER \
    --augment-K $AUG_K \
    --augment-M $AUG_M


