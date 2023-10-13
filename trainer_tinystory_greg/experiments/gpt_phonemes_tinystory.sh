# All parameters all listed in this scripts.
# Users can leave most of parameters as default value in their case.
# For parameter information, please refer to config.py

env_params="
--model_dir ./data/phonemes/model \
--ckpt gpt_512t6l6h.pt \
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
--n_layers 6 \
--n_heads 6 \
--n_embd 384 \
--max_seq_len 512 \
--dropout 0.0 \
"

optimizer_params="
--learning_rate 5e-4 \
--weight_decay 1e-1 \
--beta1 0.9 \
--beta2 0.95 \
--grad_clip 1.0 \
--decay_lr \
--decay_lr_iters 20000 \
--warmup_iters 2000 \
--min_lr 6e-5
"

trainer_params="
--init_from scratch \
--max_iters 20000 \
--batch_size 80 \
--gradient_accumulation_steps 8 \
--log_interval 1 \
--eval_interval 2000 \
--eval_iters 200 \
"

#echo "Start Training ..."
#python trainer.py $env_params $vocab_params $model_params $optimizer_params $trainer_params

echo "Start Evaluation ..."
python trainer.py $env_params $vocab_params $model_params $optimizer_params $trainer_params --eval_only --init_from resume