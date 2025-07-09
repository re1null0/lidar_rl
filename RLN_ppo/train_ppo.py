import os, sys, yaml
from pathlib import Path
import numpy as np
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from gym_interface import make_vector_env
from models import get_model
from utils import seed_everything, create_result_dir
from skrl.trainers.torch import ParallelTrainer

import multiprocessing as mp

mp.set_start_method("spawn", force=True)


class PolicyWrapper(GaussianMixin, Model):
    def __init__(self, base, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self)
        self.base = base
        self.device = next(base.parameters()).device

    def compute(self, inputs, role):
        mean, _ = self.base(inputs["states"])
        log_std = self.base.log_std.expand_as(mean)
        return mean, log_std, {}

    def forward(self, x):
        return self.base(x)[0]

    def set_mode(self, mode: str):
        self.eval() if mode == "eval" else self.train()

class ValueWrapper(DeterministicMixin, Model):
    def __init__(self, base, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self)
        self.base = base
        self.device = next(base.parameters()).device

    def forward(self, x):
        return self.base(x)[1]

    def compute(self, inputs, role):
        _, value = self.base(inputs["states"])
        return value, {}

    def set_mode(self, mode: str):
        self.eval() if mode == "eval" else self.train()

def main():
    config_path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("seed", 0))
    results_dir = create_result_dir(config["experiment_name"])
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.safe_dump(config, f)

    env = make_vector_env(config)
    device = env.device

    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, type(env.action_space)) and hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    else:
        action_dim = int(np.prod(env.action_space.shape))

    model_type = config.get("model_type", "basic_lstm").lower()
    base_model = get_model(model_type, obs_dim, action_dim, config).to(device)

    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")

    models = {
        "policy": PolicyWrapper(base_model, env.observation_space, env.action_space, device),
        "value": ValueWrapper(base_model, env.observation_space, env.action_space, device)
    }
    models["policy"].to(device)
    models["value"].to(device)

    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    cfg_agent["discount_factor"] = config.get("gamma", 0.99)
    cfg_agent["lambda"] = config.get("gae_lambda", 0.95)
    cfg_agent["learning_rate"] = config.get("learning_rate", 3e-4)
    cfg_agent["random_timesteps"] = 0
    cfg_agent["learning_epochs"] = config.get("ppo_epochs", 4)
    cfg_agent["mini_batches"] = max(1, int((config.get("rollout_steps", 1024) * env.num_envs) / config.get("batch_size", 64)))
    cfg_agent["rollouts"] = config.get("rollout_steps", 1024)
    cfg_agent["ratio_clip"] = config.get("ppo_clip", 0.2)
    cfg_agent["entropy_loss_scale"] = config.get("entropy_coef", 0.01)
    cfg_agent["experiment"]["directory"] = results_dir
    cfg_agent["experiment"]["experiment_name"] = config["experiment_name"]
    cfg_agent["experiment"]["wandb"] = config.get("wandb", {}).get("enabled", False)
    if config.get("wandb", {}).get("enabled", False):
        cfg_agent["experiment"]["writer"] = "wandb"
        cfg_agent["experiment"]["write_interval"] = config["wandb"].get("interval", 1000)
        cfg_agent["experiment"]["wandb_kwargs"] = {
            "project": config["wandb"]["project"],
            "name": config["wandb"]["run_name"],
            "tags": config["wandb"].get("tags", [])
        }
    cfg_agent["experiment"]["checkpoint_interval"] = config.get("save_interval", "auto")

    rollout_len = cfg_agent["rollouts"]
    memory_size = rollout_len * env.num_envs
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device='cuda')
    agent = PPO(models=models, memory=memory,
                observation_space=env.observation_space, action_space=env.action_space,
                device=device, cfg=cfg_agent)
    cfg_trainer = {
        "timesteps": config["total_timesteps"],
        "headless": True
    }

    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    # trainer = ParallelTrainer(env=env, agents=agent, cfg=cfg_trainer)
    trainer.train()

if __name__ == "__main__":
    main()