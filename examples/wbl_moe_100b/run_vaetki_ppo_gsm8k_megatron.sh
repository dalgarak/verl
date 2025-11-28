set -x

# Example runnable on H20 * 8

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# actor_rollout_ref_model_path="deepseek-ai/deepseek-llm-7b-chat"
# critic_model_path="deepseek-ai/deepseek-llm-7b-chat"

# DO NOT APPEND the last char in src is not '/' because it will cause error. 
export HF_ACTOR_ROLLOUT_REF_MODEL_PATH="/workspace/verl/hf_models/wbl-7b-moe-hf"
export HF_CRITIC_MODEL_PATH="/workspace/verl/hf_models/wbl-7b-moe-hf"

# DO NOT APPEND the last char in src is not '/' because it will cause error. 
export DIST_CKPT_PATH="/workspace/verl/wbl-7b-moe/iter_0130000"


python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_ACTOR_ROLLOUT_REF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    critic.optim.lr=1e-5 \
    critic.model.path=$HF_CRITIC_MODEL_PATH \
    critic.model.trust_remote_code=True \
    critic.megatron.use_dist_checkpointing=True \
    critic.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_ppo_gsm8k_math_examples' \
    trainer.experiment_name='vaetki_7b_megatron' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 $@
