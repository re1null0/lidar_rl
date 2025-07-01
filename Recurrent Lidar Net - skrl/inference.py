import os
import yaml
import torch
import numpy as np

from gym_interface import F110EnvWrapper
from models import (
    BasicLSTMNet, SpatioTemporalRLNNet, TransformerNet,
    DuelingTransformerNet, SpatioTempDuelingTransformerNet
)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def find_latest_experiment(results_dir="results"):
    exp_folders = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not exp_folders:
        raise RuntimeError("No experiment results found in 'results/'")
    latest_exp = max(exp_folders, key=os.path.getmtime)
    return latest_exp

def find_best_checkpoint(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") or f.endswith(".pth")]
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
    best_ckpt = sorted(ckpts)[-1]
    return os.path.join(checkpoint_dir, best_ckpt)

def build_model(config, obs_space, act_space, device):
    algorithm = config["algorithm"].lower()
    model_type = config.get("model_type", "basic_lstm").lower()
    obs_dim = obs_space.shape[0]
    if hasattr(act_space, "n"):
        action_dim = act_space.n
    else:
        action_dim = int(np.prod(act_space.shape))

    if algorithm == "dqn":
        # Dueling Transformer for DQN
        model = DuelingTransformerNet(obs_dim, action_dim)
    elif algorithm == "ppo":
        if model_type == "basic_lstm":
            model = BasicLSTMNet(obs_dim, action_dim)
        elif model_type == "spatiotemporal_rln":
            model = SpatioTemporalRLNNet(obs_dim, action_dim)
        elif model_type == "transformer":
            model = TransformerNet(obs_dim, action_dim)
        elif model_type == "dueling_transformer":
            model = TransformerNet(obs_dim, action_dim)
        elif model_type == "spatiotemp_dueling_transformer":
            seq_len = config.get("seq_len", 4)
            num_ranges = config.get("num_ranges", 1080)
            model = SpatioTempDuelingTransformerNet(seq_len, num_ranges, action_dim)
        else:
            model = BasicLSTMNet(obs_dim, action_dim)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return model.to(device)

def main():
    # Find latest experiment and config
    latest_exp = find_latest_experiment()
    config_path = os.path.join(latest_exp, "config.yaml")
    checkpoint_dir = os.path.join(latest_exp, "f1tenth_experiment", "checkpoints")
    best_ckpt_path = find_best_checkpoint(checkpoint_dir)

    # Load config and environment
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = F110EnvWrapper(config)
    obs, _ = env.reset()

    # Build and load model
    model = build_model(config, env.observation_space, env.action_space, device)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    print(checkpoint.keys())
    state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint

    # Remove 'base.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base."):
            new_state_dict[k[len("base."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Inference loop
    done, truncated = False, False
    total_reward = 0.0
    step = 0
    while not (done or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if config["algorithm"].lower() == "dqn":
                q_values = model(obs_tensor)
                if hasattr(env, "discrete_actions") and env.discrete_actions is not None:
                    action = int(torch.argmax(q_values, dim=1).item())
                else:
                    action = q_values.cpu().numpy()[0]
            else:  # PPO
                action_mean, _ = model(obs_tensor)
                if isinstance(env.action_space, type(env.env.action_space)) and hasattr(env, "discrete_actions") and env.discrete_actions is not None:
                    action = int(torch.argmax(action_mean, dim=1).item())
                elif hasattr(env.action_space, "n"):
                    action = int(torch.argmax(action_mean, dim=1).item())
                else:
                    action = action_mean.cpu().numpy()[0]
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        if hasattr(env.env, "render"):
            env.env.render()

    print(f"Episode finished in {step} steps. Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()