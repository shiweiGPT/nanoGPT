import os
import math
import argparse
from functools import partial

import torch

#import sys 
#sys.path.append("..") 
from modelLlama2 import Transformer, ModelArgs # Llama2
from modelGPT2 import GPT, GPTConfig

# ---------------------------------------------------------------------
# 1. Parse arguments

def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--n_layers'
            # e.g., param_config = 'type', 'default', 'help' and 'dest'...
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)

def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config['dest']:
                namespace.__getattribute__(param_config['dest'])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }

# ---------------------------------------------------------------------
# 2. Setup Environment Parameters
# TODO: Add ddp in the setup_env function for multi-processing

def setup_env(env_params):
    if not torch.cuda.is_available():
        env_params['device'] = 'cpu'
    elif env_params['device'] is None:
        env_params['device'] = 'cuda'

# ---------------------------------------------------------------------
# 3. Setup Model and Optimizer Parameters

def setup_model_and_optimizer(model_params, optimizer_params, init_from, ckpt_path, vocab_size, device):
    assert init_from.lower() in ['scratch', 'resume'], \
        f'Error: Only support training model from scratch or reload model from a ckpt, but get {init_from}'
    assert model_params['model_type'].lower() in ['gpt', 'llama'], \
        f'Error: Only support gpt and llama. Unknow model type, but get {init_from}'

    if model_params['model_type'] == 'gpt':
        model_args = dict(n_layer=model_params['n_layers'], 
                              n_head=model_params['n_heads'], 
                              n_embd=model_params['n_embd'], 
                              block_size=model_params['max_seq_len'],
                              bias=model_params['bias'], 
                              vocab_size=vocab_size, 
                              dropout=model_params['dropout'])
    else:
        model_args = dict(n_layers=model_params['n_layers'], 
                              n_heads=model_params['n_heads'], 
                              dim=model_params['n_embd'], 
                              max_seq_len=model_params['max_seq_len'],
                              vocab_size=vocab_size, 
                              dropout=model_params['dropout'], 
                              n_kv_heads=model_params['n_kv_heads'], 
                              multiple_of=model_params['multiple_of'])

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        if model_params['model_type'] == 'gpt':
            model_config = GPTConfig(**model_args)
            model = GPT(model_config)
        else: #llama
            model_config = ModelArgs(**model_args)
            model = Transformer(modle_config)
        iter_num = 0
        best_val_loss = 1e9
        checkpoint=None
    else: # resume
        assert os.path.exists(ckpt_path), f'ckeckpoint: {ckpt_path} does not exist.'
        print(f"Resuming training from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        if model_params['model_type'] == 'gpt':
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            model_config = GPTConfig(**model_args)
            model = GPT(model_config)
        else: # llama
            for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
                model_args[k] = checkpoint_model_args[k]
            model_config = ModelArgs(**model_args)
            model = Transformer(modle_config)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)
            
    model.to(device)
    optimizer = model.configure_optimizers(
                              weight_decay=optimizer_params['weight_decay'], 
                              learning_rate=optimizer_params['learning_rate'], 
                              betas=(optimizer_params['beta1'], optimizer_params['beta2']), 
                              device_type=device)
    if init_from == 'resume' and checkpoint is not None and "optimizer" in checkpoint:
        print((f"Resuming optimizer from {ckpt_path}"))
        optimizer.load_state_dict(checkpoint["optimizer"])

    checkpoint=None

    return model, optimizer, model_args, iter_num, best_val_loss


if __name__ == '__main__':
    params = get_params({'vocab_params': {
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
    },})

    print(params)