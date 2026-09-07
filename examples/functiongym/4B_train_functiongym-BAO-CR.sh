#!/bin/bash
# =============================================================================
# BAO RL training on Function-Gym (Qwen3-4B, single node, 8 GPUs).
#
# Usage:
#   1) Download the SFT checkpoints (see README) and set SFT_CKPT_DIR to the
#      Function-Gym checkpoint folder (holding config.json + *.safetensors).
#   2) Export your Weights & Biases key:  export WANDB_API_KEY=<key>
#      (or set trainer.logger=[console] below to disable W&B).
#   3) Run:  bash examples/functiongym/4B_train_functiongym-BAO-CR.sh
#
# Any variable below can be overridden from the environment, e.g.:
#   SFT_CKPT_DIR=/path/to/ckpt WANDB_API_KEY=xxxx bash .../4B_train_functiongym-BAO-CR.sh
# =============================================================================

set -x
export RAY_DISABLE_DASHBOARD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
ulimit -n 65535

# ---- Paths ------------------------------------------------------------------
# Repo root is auto-detected from this script's location; override PROJECT_DIR to change it.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_PATH="$PROJECT_DIR/examples/functiongym/config"

# Path to the downloaded Function-Gym SFT checkpoint (REQUIRED).
SFT_CKPT_DIR="${SFT_CKPT_DIR:?Please set SFT_CKPT_DIR to your downloaded BAO Function-Gym SFT checkpoint directory}"

EXP_NAME="${EXP_NAME:-Function-Gym-BAO-4B-overlengthpen-0.05-repeat-pen-0.005}"

# ---- Weights & Biases -------------------------------------------------------
# Get a key from https://wandb.ai/authorize . For a self-hosted instance also set WANDB_BASE_URL.
# To disable W&B, set trainer.logger=[console] below and ignore WANDB_API_KEY.
export WANDB_API_KEY="${WANDB_API_KEY:?Please export WANDB_API_KEY, or set trainer.logger=[console] to disable wandb}"
# export WANDB_BASE_URL="https://api.wandb.ai"

# ---- User simulator ---------------------------------------------------------
# Used by Turtle-Gym / Telepathy-Gym only. Function-Gym uses rule-based rewards
# and ignores these, so the defaults below are fine for this script.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export MULTITURN_MODEL_NAME="${MULTITURN_MODEL_NAME:-gpt-4o}"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_multiturn' \
    algorithm.adv_estimator=grpo_multiturn \
    algorithm.gamma=0.8 \
    data.train_batch_size=128 \
    data.max_prompt_length=1152 \
    data.max_response_length=7000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$SFT_CKPT_DIR" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.max_turns=16 \
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/interact_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="R2G" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum" \
    actor_rollout_ref.rollout.multi_turn.penalize_over_length_coef=0.05 \
    actor_rollout_ref.rollout.multi_turn.repeated_question_pen_coef=0.005 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='UserRL-functiongym-4B' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    data.train_files=$PROJECT_DIR/data/function_multiturn/train.parquet \
    data.val_files=$PROJECT_DIR/data/function_multiturn/test.parquet \
    trainer.validation_data_dir="${PROJECT_DIR}/validation_trajectory/functiongym/${EXP_NAME}" \
    trainer.total_epochs=30 $@
