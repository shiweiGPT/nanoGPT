'''
Command-line arguments with their default values. Inspired from facebook-adaptive-attention reop.
'''

PARAMS_CONFIG = {
    # env-specific
    'env_params': {
        '--model_dir': {
            'type': str,
            'default': './out',
            'help': 'The path for checkpoint saving',
            'dest': 'model_dir'
        },
        '--ckpt': {
            'type': str,
            'default': 'ckpt.pt',
            'help': 'The name for checkpoint',
            'dest': 'ckpt'
        },
        '--device': {
            'type': str,
            'default': 'cuda',
            'help': 'The device to run the training',
            'dest': 'device'
        },
        '--dtype': {
            'type': str,
            'default': 'bfloat16',
            'help': 'Data type. could be float32|bfloat16|float16',
            'dest': 'dtype'
        },
    },
    # vocab-specific
    'vocab_params': {
        '--vocab_data': {
            'type': str,
            'default': './data/phonemes',
            'help': 'The path for dataset',
            'dest': 'vocab_data'
        },
        '--vocab_size': {
            'type': int,
            'default': 2048,
            'help': 'The vocabulary size of the trained tokenizer',
            'dest': 'vocab_size'
        },
        '--vocab_source': {
            'type': str,
            'default': 'custom',
            'help': 'Only support custom trained tokenizer for now' 
                    'TODO: support tiktoken for gpt and llama tokenizer',
            'dest': 'vocab_source'
        },
    },
    # model-specific
    'model_params': {
        '--model_type': {
            'type': str,
            'default': 'gpt',
            'help': 'Only support gpt and llama now for nanoGPT',
            'dest': 'model_type'
        },
        '--n_layers': {
            'type': int,
            'default': 8,
            'help': 'Number of layers',
            'dest': 'n_layers'
        },
        '--n_heads': {
            'type': int,
            'default': 8,
            'help': 'Number of attention heads',
            'dest': 'n_heads'
        },
        '--n_embd': {
            'type': int,
            'default': 512,
            'help': 'The length of attention feature, generally set as 64*n_heads'
                    'For feed-forward layer, the feature size is 4*n_embd',
            'dest': 'n_embd'
        },
        '--max_seq_len': {
            'type': int,
            'default': 512,
            'help': 'The length of sequence for traning'
                    'Also named as block_size',
            'dest': 'max_seq_len'
        },
        '--dropout': {
            'type': float,
            'default': 0.2,
            'help': 'dropout value',
            'dest': 'dropout'
        },
        '--bias': {
            'action': 'store_true',
            'default': False,
            'help': 'Using bias in linear layer or not.'
                    'TODO: Only valid for gpt model for now. Support this configuration in llama',
            'dest': 'bias'
        },
        '--n_kv_heads': {
            'type': int,
            'default': 8,
            'help': 'The number of attention head for Key and Value. Inspired from Group-Query Attention'
                    'TODO: only support llama for now. maybe support gpt in future',
            'dest': 'n_kv_heads'
        },
        '--multiple_of': {
            'type': int,
            'default': 32,
            'help': 'Feed-forward hidden layer size will be multiple of in llama',
            'dest': 'multiple_of'
        },
    },
    # optimizer-specific
    'optimizer_params': {
        '--optimizer': {
            'type': str,
            'default': 'adamw',
            'help': 'TODO: only support adamw for now. support others in furture',
            'dest': 'optimizer'
        },
        '--learning_rate': {
            'type': float,
            'default': 5e-4,
            'help': 'Learning rate',
            'dest': 'learning_rate'
        },
        '--weight_decay': {
            'type': float,
            'default': 0.1,
            'help': 'weight deacy in adamw optimizer',
            'dest': 'weight_decay'
        },
        '--beta1': {
            'type': float,
            'default': 0.9,
            'help': 'beta 1 in adamw optimizer',
            'dest': 'beta1'
        },
        '--beta2': {
            'type': float,
            'default': 0.95,
            'help': 'beta 2 in adamw optimizer',
            'dest': 'beta2'
        },
        '--grad_clip': {
            'type': float,
            'default': 1.0,
            'help': 'Clip gradient of each layer parameters by a given value',
            'dest': 'grad_clip'
        },
        '--decay_lr': {
            'action': 'store_true',
            'default': False,
            'help': 'Whether to decay the learning rate',
            'dest': 'decay_lr'
        },
        '--decay_lr_iters': {
            'type': int,
            'default': 20000,
            'help': 'should be ~= max_iters per Chinchilla',
            'dest': 'decay_lr_iters'
        },
        '--warmup_iters': {
            'type': int,
            'default': 2000,
            'help': 'Linearly increase the learning rate from 0 during warming up iterations',
            'dest': 'warmup_iters'
        },
        '--min_lr': {
            'type': float,
            'default': 6e-5,
            'help': 'Linearly increase the learning rate from 0 during warming up iterations'
                    'Minimum learning rate, should be ~= learning_rate/10 per Chinchilla',
            'dest': 'min_lr'
        },
    },
    # trainer-specific
    'trainer_params' : {
        '--init_from': {
            'type': str,
            'default': 'scratch',
            'help': 'Initialize the training from [scratch, resume]'
                    'TODO: suuport pretrained model such as gpt, llama',
            'dest': 'init_from'
        },
        '--eval_only': {
            'action': 'store_true',
            'default': False,
            'help': 'Only evaluate the trained model',
            'dest': 'eval_only'
        },
        '--max_iters': {
            'type': int,
            'default': 20000,
            'help': 'Number of iterations to train',
            'dest': 'max_iters'
        },
        '--batch_size': {
            'type': int,
            'default': 80,
            'help': 'Batch size',
            'dest': 'batch_size'
        },
        '--gradient_accumulation_steps': {
            'type': int,
            'default': 8,
            'help': 'Used to simulate large batch size',
            'dest': 'gradient_accumulation_steps'
        },
        '--log_interval': {
            'type': int,
            'default': 1,
            'help': 'Print train loss every #log_interval steps',
            'dest': 'log_interval'
        },
        '--eval_interval': {
            'type': int,
            'default': 1000,
            'help': 'Evaluate the loss on train/val sets' 
                    'and write checkpoints every #eval_interval steps',
            'dest': 'eval_interval'
        },
        '--eval_iters': {
            'type': int,
            'default': 200,
            'help': 'The number of iteration when evaluating the loss on val set',
            'dest': 'eval_iters'
        },
        '--always_save_ckpt': {
            'action': 'store_true',
            'default': False,
            'help': 'If True, always save a ckpt after each evalutaion',
            'dest': 'always_save_ckpt'
        },
    },
    # rl-specific
    'rl_params': {
        '--agent_type': {
            'type': str,
            'default': 'ppo',
            'help': 'agent type',
            'dest': 'agent_type'
        },
        '--rl_timesteps': {
            'type': int,
            'default': 100,
            'help': 'the training step for rl agent',
            'dest': 'rl_timesteps'
        },
        '--ppl_factor': {
            'type': float,
            'default': 0.5,
            'help': 'scaling factor for ppl_reward. fall in range [0, 1]',
            'dest': 'ppl_factor'
        },
        '--obs_n_layers': {
            'type': int,
            'default': 12,
            'help': 'set the max layer number for observation space',
            'dest': 'obs_n_layers'
        },
        '--obs_n_heads': {
            'type': int,
            'default': 12,
            'help': 'set the max head number for observation space',
            'dest': 'obs_n_heads'
        },
        '--obs_n_embd_per_head': {
            'type': int,
            'default': 128,
            'help': 'set the max embedding size per head for observation space',
            'dest': 'obs_n_embd_per_head'
        },
        '--n_layers_step': {
            'type': int,
            'default': 1,
            'help': 'fixed step to update layer configuration',
            'dest': 'n_layers_step'
        },
        '--n_heads_step': {
            'type': int,
            'default': 1,
            'help': 'fixed step to update head configuration',
            'dest': 'n_heads_step'
        },
        '--n_embd_per_head_step': {
            'type': int,
            'default': 1,
            'help': 'fixed step to update embedding configuration',
            'dest': 'n_embd_per_head_step'
        },
    },
    # wandb-specific for training visulization
    # default using Pytorch Tensorborad
    'wandb_params': {
        '--wandb_log': {
            'action': 'store_true',
            'default': False,
            'help': 'Enable wandb',
            'dest': 'wandb_log'
        },
        '--wandb_project': {
            'type': str,
            'default': 'nanoGPT',
            'help': 'Wandb project name',
            'dest': 'wandb_project'
        },
        '--wandb_run_name': {
            'type': str,
            'default': 'run_nanoGPT',
            'help': 'Wandb Log name in the current project',
            'dest': 'wandb_run_name'
        },
    },
}