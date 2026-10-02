#!/usr/bin/env bash
# V1 Megatron + vLLM, bypass REINFORCE with token TIS and score centering.
# Prepare GSM8K parquet files first. This is a GPU smoke entrypoint, not a benchmark result.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-data/train.parquet}
TEST_FILE=${TEST_FILE:-data/test.parquet}

LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra megatron")
fi

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    model_engine=megatron \
    trainer.use_v1=true \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.use_fused_kernels=false \
    actor_rollout_ref.actor.megatron.use_remove_padding=true \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.score_centering_chunk_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.topk_log_probs=128 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=4 \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=true \
    algorithm.rollout_correction.loss_type=reinforce \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.score_centering=true \
    trainer.logger='[console]' \
    trainer.project_name=score_centering_megatron \
    trainer.experiment_name=qwen2_5_0_5b_token_tis_sc \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=2 \
    "${RAY[@]}" \
    "$@"
