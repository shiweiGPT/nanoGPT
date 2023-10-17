'''
Software reinforcement learning loop to explore the LLM structures. 
Inspired from github repo klei22/RL-Templates-2.
'''
import math
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trainer import train_launcher
from config_util import get_params
from config import PARAMS_CONFIG

def calculate_model_size(n_layers, n_heads, n_embd_per_head):
    size_per_layer = 12*((n_heads * n_embd_per_head)**2)
    return size_per_layer * n_layers

# TODO
def target_function(ppl, ppl_factor, model_size, max_model_size):
    if model_size > max_model_size:
        print('LOG SW: ------------------ Penalty Model Size ------------------')
        return (max_model_size/model_size) - 1
    if ppl > 3.0:
        print('LOG SW: ------------------ Penalty Perplexity ------------------')
        return 1/ppl - 1
    s_reward = 1-model_size/max_model_size
    p_reward = (1/ppl)
    reward = ppl_factor * p_reward + (1-ppl_factor) * s_reward
    #reward = math.sqrt(p_reward * s_reward)
    print('LOG SW: ------------------ Reward model size ------------------')
    return reward

#def target_function(ppl, ppl_factor, model_size, max_model_size):
#    return (1-model_size/max_model_size)/ppl

class NanoEnv(gym.Env):
    def __init__ (self, env_params, vocab_params, model_params, 
                  optimizer_params, trainer_params, rl_params, wandb_params):
        super(NanoEnv, self).__init__()

        # 1. Record initial setting
        self.env_params = env_params
        self.vocab_params = vocab_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.wandb_params = wandb_params
        self.rl_params = rl_params

        # 2. Setup observation spaces for model structure. For now, just support n_layers, n_heads, n_embd
        # TODO: Add model configuration options, like n_kv_head for llama.
        self.observation_space = spaces.Box(low=1, high=np.inf, shape=(3,), dtype=np.int32)
        #self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,))

        # 3. Setup action spaces
        #self.action_space = spaces.MultiDiscrete([2, 2, 2])
        #self.action_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([2, rl_params['n_layers_step'], 2, rl_params['n_heads_step'], 2, rl_params['n_embd_per_head_step']])

        # 4. Initialize model state for the first rl step
        self.n_layers = self.model_params['n_layers']
        self.n_heads = self.model_params['n_heads']
        assert self.model_params['n_embd'] % self.model_params['n_heads'] == 0, \
            f"n_embd {self.model_params['n_embd']} should be a multiple of heads {self.model_params['n_heads']}"
        self.n_embd_per_head = self.model_params['n_embd'] // self.model_params['n_heads']

        max_n_layers = self.rl_params['obs_n_layers']
        max_n_heads = self.rl_params['obs_n_heads']
        max_n_embd_per_head = self.rl_params['obs_n_embd_per_head']
        self.rl_params['max_model_size'] = calculate_model_size(max_n_layers, max_n_heads, max_n_embd_per_head)
        self.rl_params['n_layers'] = self.model_params['n_layers']
        self.rl_params['n_heads'] = self.model_params['n_heads']
        self.rl_params['n_embd_per_head'] = self.model_params['n_embd'] // self.model_params['n_heads']
    
    def step(self, action):
        print('LOG SW: Action: ', action)
        # 1. Take next step with the actions
        # Done: Update model structure with dynamical steps. For now, just change the state linearly with fixed steps.
        '''
        self.n_layers = max(1, (self.n_layers - self.rl_params['n_layers_step'])) \
                                    if action[0] == 1 else (self.n_layers + self.rl_params['n_layers_step'])
        self.n_heads = max(1, (self.n_heads - self.rl_params['n_heads_step'])) \
                                    if action[1] == 1 else (self.n_heads + self.rl_params['n_heads_step'])
        self.n_embd_per_head = max(1, (self.n_embd_per_head - self.rl_params['n_embd_per_head_step'])) \
                                    if action[2] == 1 else (self.n_embd_per_head + self.rl_params['n_embd_per_head_step'])
        action = [int(round(i)) for i in action]
        print('LOG SW: Round Action: ', action)
        self.n_layers = max(1, (self.n_layers+action[0]))
        self.n_heads = max(1, (self.n_heads+action[1]))
        self.n_embd_per_head = max(1, (self.n_embd_per_head+action[2]))
        '''

        layers_dir, layers_step, heads_dir, heads_step, embd_dir, embd_step = action
        self.n_layers = max(self.n_layers + (layers_dir*2-1) * layers_step, np.int64(1)).item()
        self.n_heads = max(self.n_heads + (heads_dir*2-1) * heads_step, np.int64(1)).item()
        self.n_embd_per_head = max(self.n_embd_per_head + (embd_dir*2-1) * embd_step, np.int64(1)).item()

        #self.n_layers = np.clip(self.n_layers, 1, np.inf).item()
        #self.n_heads = np.clip(self.n_heads, 1, np.inf).item()
        #self.n_embd_per_head = np.clip(self.n_embd_per_head, 1, np.inf).item()

        # 2. Train the NanoGPT with the new configuration
        self.model_params['n_layers'] = self.n_layers
        self.model_params['n_heads'] = self.n_heads
        self.model_params['n_embd'] = self.n_embd_per_head * self.n_heads
        val_loss = train_launcher(self.env_params, self.vocab_params, self.model_params, 
                                 self.optimizer_params, self.trainer_params, self.wandb_params)
        val_ppl = math.exp(val_loss)
        # 3. Generate the reward
        model_size = calculate_model_size(self.n_layers, self.n_heads, self.n_embd_per_head)
        reward = target_function(val_ppl, self.rl_params['ppl_factor'], model_size, self.rl_params['max_model_size'])

        print('LOG SW: Reward, PPL and Model Size: ', reward, val_ppl, self.n_layers, self.n_heads, self.n_embd_per_head * self.n_heads)

        # 4. TODO: Kill the rl loop when the model meets a certain requirement.
        # Just set "done" to False here, and kill the rl training loop with the max_steps.
        done = False

        return np.array([self.n_layers, self.n_heads, self.n_embd_per_head]), reward, done, {}

    def reset(self):
        # TODO: Initialize the model with randam seed. For now, just restore the model from config list
        self.n_layers, self.n_heads, self.n_embd_per_head_step = \
            self.rl_params['n_layers'], self.rl_params['n_heads'], self.rl_params['n_embd_per_head_step']
        return np.array([self.n_layers, self.n_heads, self.n_embd_per_head_step])

def get_agent(agent_type, env):
    if agent_type.lower() == "ppo":
        return PPO('MlpPolicy', env, ent_coef=0.001, verbose=1, n_steps=1, tensorboard_log="./ppo_tensorboard/")
        #return PPO('MultiInputPolicy', env, ent_coef=0.01, verbose=1, tensorboard_log="./ppo_tensorboard/")
    elif agent_type.lower() == "a2c":
        print('find a2c')
        return A2C('MlpPolicy', env, ent_coef=0.001, verbose=1, n_steps=1, tensorboard_log="./a2c_tensorboard/")
        #return A2C('MultiInputPolicy', env, ent_coef=0.01, verbose=1, tensorboard_log="./a2c_tensorboard/")
    else:
        raise ValueError(f"Unknown model name {agent_type}, please choose either 'ppo' or 'a2c'")

def rl_trainer(env_params, vocab_params, model_params, optimizer_params, trainer_params, rl_params, wandb_params):
    # 1. Initialize the rl environment and agent
    np.random.seed(7)
    env = NanoEnv(env_params, vocab_params, model_params, 
                  optimizer_params, trainer_params, rl_params, wandb_params)
    env = DummyVecEnv([lambda: env])
    
    # 2. Initialize the agent
    agent = get_agent(rl_params['agent_type'], env)

    # 3. Start rl loop
    print('RL Configuration: ', rl_params['rl_timesteps'])
    #agent.set_random_seed(7)
    agent.learn(total_timesteps=rl_params['rl_timesteps'], log_interval=1)
    print(f"Final RL Results: n_layers {env.envs[0].n_layers}, n_heads {env.envs[0].n_heads} \
            and e_embd {env.envs[0].n_heads * env.envs[0].n_embd_per_head}")
    
    agent.save('agent_model')

if __name__ == "__main__":
    rl_trainer(**get_params(params_config=PARAMS_CONFIG))