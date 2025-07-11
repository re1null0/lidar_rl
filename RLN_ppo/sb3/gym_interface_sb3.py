import os, sys, yaml
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from gym_interface import F110EnvWrapper

def make_vec_env_sb3(cfg: dict, n_envs: int = 1, use_subproc: bool = False):
    seed = cfg.get("seed", 0)
    def make_one(rank):
        def _init():
            env = F110EnvWrapper(cfg, seed + rank)
            return RecordEpisodeStatistics(env)
        return _init

    if n_envs == 1:
        env = DummyVecEnv([make_one(0)])
    else:
        env = SubprocVecEnv([make_one(i) for i in range(n_envs)])
    return VecNormalize(env, norm_obs=True, norm_reward=False)
