import os
import yaml
import torch
import numpy as np
import gymnasium as gym
import time
from f1tenth_gym.envs.f110_env import F110Env

from models import get_model
from train_ppo import PolicyWrapper

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

def process_observation(obs_dict, config):
    """Convert raw observation dict to model input format"""
    obs_vec = []
    
    # Extract speed if configured
    if config.get("include_velocity_in_obs", True):
        try:
            # Update: handle F1TENTH gym observation structure
            speed = obs_dict['ego_idx']['velocity'][0]  # Forward velocity from ego vehicle
            obs_vec.append(speed)
        except (KeyError, IndexError):
            try:
                # Alternative: try standard state vector
                speed = obs_dict['agent_0']['std_state'][3]  # velocity component
                obs_vec.append(speed)
            except (KeyError, IndexError):
                obs_vec.append(0.0)
    
    # Extract LiDAR
    if config["lidar"]["enabled"]:
        try:
            # Update: handle different possible LiDAR keys
            if 'observations' in obs_dict:
                scan = obs_dict['observations']['scan']
            elif 'scan' in obs_dict:
                scan = obs_dict['scan']
            elif 'agent_0' in obs_dict and 'scan' in obs_dict['agent_0']:
                scan = obs_dict['agent_0']['scan']
            else:
                raise KeyError("Could not find LiDAR scan in observation dict")
                
            if config["lidar"]["downsample"]:
                scan = scan[::10]  # Downsample to 108 points
            obs_vec.extend(scan)
        except KeyError as e:
            print(f"Warning: Could not extract LiDAR scan. Observation dict keys: {obs_dict.keys()}")
            print(f"Full observation dict: {obs_dict}")
            raise e
    
    return np.array(obs_vec, dtype=np.float32)

def main():
    # Find latest experiment and config
    latest_exp = find_latest_experiment()
    config_path = os.path.join(latest_exp, "config.yaml")
    checkpoint_dir = os.path.join(latest_exp, "f1tenth_experiment", "checkpoints")
    best_ckpt_path = find_best_checkpoint(checkpoint_dir)

    # Load config
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get absolute path to map file
    map_path = "/home/saichand/ros2_ws/src/RL_RecurrentLidarNet/RLN_ppo/maps/levine.yaml"
    
    # Create F1TENTH environment with proper config
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator_timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "ks",  # "ks", "st", "mb"
            "observation_config": {"type": "direct"},
            "params": F110Env.f1tenth_vehicle_params(),
            "reset_config": {"type": "rl_random_static"},
            "map_scale": 1.0,
            "enable_rendering": 1,
            "enable_scan": 1,
            "lidar_num_beams": 1080,
            "compute_frenet": 0,
            "renderer_config": {
                "background_color": [255, 255, 255]
            },
            "max_laps": 5,  # Increase number of laps
            "terminate_on_collision": False,  # Don't terminate on collision
        },
        render_mode="human"  # Change to human for slower visualization
    )

    # Initialize environment
    obs_dict, info = env.reset()
    env.render()  # Initial render

    # Process observation
    processed_obs = process_observation(obs_dict, config)
    obs_dim = len(processed_obs)
    action_dim = 2  # [steering, velocity]

    # Create observation/action spaces
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

    # Build base model
    model_type = config.get("model_type", "basic_lstm").lower()
    base_model = get_model(model_type, obs_dim, action_dim, config)
    base_model = base_model.to(device)

    # Wrap with PolicyWrapper
    model = PolicyWrapper(base_model, obs_space, action_space, device)

    # Load checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    if "policy" in checkpoint:
        model.load_state_dict(checkpoint["policy"])
    else:
        raise RuntimeError("Checkpoint does not contain 'policy' weights")

    model.eval()

    # Inference loop with multiple episodes
    num_episodes = 5  # Number of episodes to run
    for episode in range(num_episodes):
        obs_dict, info = env.reset()
        env.render()  # Initial render
        
        done = False
        total_reward = 0.0
        step = 0
        start_time = time.time()

        while not done:
            # Process observation
            processed_obs = process_observation(obs_dict, config)
            obs_tensor = torch.tensor(processed_obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Get action from policy
            with torch.no_grad():
                action_mean, _, _ = model.compute({"states": obs_tensor}, "eval")
                action = action_mean.cpu().numpy()[0]

            # Step environment
            obs_dict, reward, done, truncated, info = env.step(np.array([action]))
            total_reward += reward
            step += 1

            # Add small delay for visualization
            time.sleep(0.01)  # 10ms delay per step
            
            # Render environment
            frame = env.render()

            # Optional: Break if episode is too long
            if step > 5000:  # Max steps per episode
                break

        elapsed_time = time.time() - start_time
        print(f"Episode {episode + 1} finished in {step} steps. Total reward: {total_reward:.2f}")
        print(f"Episode elapsed time: {elapsed_time:.2f}s")

    env.close()

if __name__ == "__main__":
    main()