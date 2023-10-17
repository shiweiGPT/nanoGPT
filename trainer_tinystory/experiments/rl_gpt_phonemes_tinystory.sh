# All parameters all listed in this scripts.
# Users can leave most of parameters as default value in their case.
# For parameter information, please refer to config.py

env_params="
--model_dir ./data/phonemes/rl_model \
--ckpt gpt_rl.pt \
--device cuda \
--dtype bfloat16 \
"

vocab_params="
--vocab_data ./data/phonemes \
--vocab_size 2048 \
--vocab_source custom
"

model_params="
--model_type gpt \
--n_layers 2 \
--n_heads 2 \
--n_embd 128 \
--max_seq_len 256 \
--dropout 0.0 \
"

optimizer_params="
--learning_rate 5e-4 \
--weight_decay 1e-1 \
--beta1 0.9 \
--beta2 0.95 \
--grad_clip 1.0 \
--decay_lr \
--decay_lr_iters 2000 \
--warmup_iters 100 \
--min_lr 6e-5
"

trainer_params="
--init_from scratch \
--max_iters 2000 \
--batch_size 40 \
--gradient_accumulation_steps 1 \
--log_interval 1 \
--eval_interval 500 \
--eval_iters 200 \
"

rl_params="
--agent_type a2c \
--rl_timesteps 200 \
--ppl_factor 0.8 \
--obs_n_layers 6 \
--obs_n_heads 6 \
--obs_n_embd_per_head 64 \
--n_layers_step 3 \
--n_heads_step 3 \
--n_embd_per_head_step 10
"

#echo "Start Training ..."
python rl_nanogpt.py $env_params $vocab_params $model_params $optimizer_params $trainer_params $rl_params