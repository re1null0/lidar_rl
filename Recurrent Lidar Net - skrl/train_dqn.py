import os, sys, yaml
from pathlib import Path
import numpy as np
import torch
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, DeterministicMixin

from gym_interface import make_vector_env
from models import get_model
from utils import seed_everything, create_result_dir

class DuelingQNet(DeterministicMixin, Model):
    def __init__(self, backbone, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self)
        self.backbone = backbone.to(device)
        self.device = device

    def compute(self, inputs, role):
        q_values = self.backbone(inputs["states"])
        return q_values, {}

    def forward(self, x):
        return self.backbone(x)

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
    num_actions = env.action_space.n if hasattr(env.action_space, 'n') else int(np.prod(env.action_space.shape))

    q_backbone = get_model(config.get("model_type", "basic_lstm").lower(), obs_dim, num_actions, config)
    target_backbone = get_model(config.get("model_type", "basic_lstm").lower(), obs_dim, num_actions, config)

    models = {
        "q_network": DuelingQNet(q_backbone, env.observation_space, env.action_space, device),
        "target_q_network": DuelingQNet(target_backbone, env.observation_space, env.action_space, device)
    }
    models["q_network"].to(device)
    models["target_q_network"].to(device)

    memory = RandomMemory(memory_size=config["replay_buffer_size"], num_envs=env.num_envs, device=device)

    cfg_agent = DQN_DEFAULT_CONFIG.copy()
    cfg_agent["discount_factor"] = config.get("gamma", 0.99)
    cfg_agent["batch_size"] = config.get("batch_size", 64)
    cfg_agent["exploration"]["initial_epsilon"] = config.get("epsilon_start", 1.0)
    cfg_agent["exploration"]["final_epsilon"] = config.get("epsilon_end", 0.05)
    cfg_agent["exploration"]["timesteps"] = config.get("epsilon_decay", 10000)
    cfg_agent["experiment"]["directory"] = results_dir
    cfg_agent["experiment"]["experiment_name"] = config["experiment_name"]
    cfg_agent["experiment"]["wandb"] = config.get("wandb", {}).get("enabled", False)
    if config.get("wandb", {}).get("enabled", False):
        cfg_agent["experiment"]["writer"] = "wandb"
        cfg_agent["experiment"]["write_interval"] = config.get("save_interval", "auto")
        cfg_agent["experiment"]["wandb_kwargs"] = {
            "project": config["wandb"]["project"],
            "name": config["wandb"]["run_name"],
            "tags": config["wandb"].get("tags", [])
        }
    cfg_agent["experiment"]["checkpoint_interval"] = config.get("save_interval", "auto")

    agent = DQN(models=models, memory=memory,
                observation_space=env.observation_space, action_space=env.action_space,
                device=device, cfg=cfg_agent)

    cfg_trainer = {
        "timesteps": config["total_timesteps"],
        "headless": True
    }
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    trainer.train()

if __name__ == "__main__":
    main()