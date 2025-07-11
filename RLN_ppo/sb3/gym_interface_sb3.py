import yaml
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gym_interface import F110EnvWrapper  

def make_vec_env(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_envs = cfg.get("n_envs", 1)
    seed   = cfg.get("seed", 0)

    def make_one(rank):
        def _init():
            env = F110EnvWrapper(cfg, seed + rank)
            env = RecordEpisodeStatistics(env)
            return env
        return _init

    if n_envs == 1:
        env = F110EnvWrapper(cfg, seed)
        env = RecordEpisodeStatistics(env)
        vec = DummyVecEnv([lambda: env])
    else:
        vec = SubprocVecEnv([make_one(i) for i in range(n_envs)])

    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    return vec
