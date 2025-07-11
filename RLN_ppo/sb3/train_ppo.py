import os
import sys
import yaml
import random
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_interface_sb3 import make_vec_env_sb3


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_result_dir(experiment_name: str) -> str:
    results_dir = os.path.join("results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def main(config_path: str = None):
    # Load configuration
    config_path = config_path or (sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Seed
    seed_everything(config.get("seed", 0))

    # Prepare results
    results_dir = create_result_dir(config["experiment_name"])
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.safe_dump(config, f)

    # Create vectorized env
    env = make_vec_env_sb3(
        config,
        n_envs=config.get("n_envs", 1),
        use_subproc=config.get("use_subproc", False)
    )

    # Instantiate Recurrent PPO with LSTM policy
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("rollout_steps", 1024),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("ppo_epochs", 4),
        gamma=config.get("gamma", 0.99),
        ent_coef=config.get("entropy_coef", 0.01),
        clip_range=config.get("ppo_clip", 0.2),
        seed=config.get("seed", 0),
        tensorboard_log=os.path.join(results_dir, "tensorboard")
    )

    # Checkpoint callback
    checkpoint_freq = config.get("save_interval", 100000)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=results_dir,
        name_prefix="ppo_model"
    )

    # Train
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=checkpoint_callback
    )

    # Final save
    model.save(os.path.join(results_dir, "final_model"))
    env.close()


if __name__ == "__main__":
    main()
