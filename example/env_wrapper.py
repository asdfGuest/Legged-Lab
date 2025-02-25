import torch as th
from simple_rl.env import BaseEnv as SimpleRlEnv
from legged_lab.base.env import BaseEnv as LeggedEnv


class LeggedEnvWrapper(SimpleRlEnv) :
    def __init__(self, env:LeggedEnv, reward_scale:float=1.0) :
        self.env = env
        self.device = env.device
        self.n_env = env.num_envs
        self.n_obs = env.n_obs * env.n_history + env.n_privileged_obs
        self.n_action = env.n_action
        self.reward_scale = reward_scale
    
    def _obs(self, obs_pack) :
        obs_h, privileged_obs = obs_pack['policy']
        return th.cat([obs_h.flatten(1), privileged_obs], dim=-1)
    
    def reset(self) :
        obs_pack, info = self.env.reset()
        return self._obs(obs_pack), info
    
    def step(self, action) :
        obs_pack, reward, terminated, truncated, info = self.env.step(action)
        reward = reward.view(self.n_env,1) * self.reward_scale
        terminated = terminated.view(self.n_env,1)
        truncated = truncated.view(self.n_env,1)
        return self._obs(obs_pack), reward, terminated, truncated, info
    
    def close(self) :
        return self.env.close()
