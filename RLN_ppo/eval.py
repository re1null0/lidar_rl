import sys, yaml
import os, sys, yaml
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gym_interface import make_vector_env
from models import BasicLSTMNet, SpatioTemporalRLNNet, TransformerNet, DuelingTransformerNet

# Load config and model from specified results directory
if len(sys.argv) < 2:
    print("Usage: python eval.py <results_dir>")
    sys.exit(0)
results_dir = sys.argv[1]
config_path = f"{results_dir}/config.yaml"
model_path = f"{results_dir}/best_agent.pt"
# For DQN, adjust model filename if saved differently
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Config file not found in {results_dir}")
    sys.exit(0)
algo = config.get("algorithm", "ppo").lower()
# Use 1 environment for evaluation (single car)
config['n_envs'] = 1
# Disable randomization and noise for evaluation
config['domain_randomization'] = False
config['sensor_noise'] = {'lidar': 0.0, 'speed': 0.0}
# Create environment
env = make_vector_env(config)
obs_space = env.observation_space
act_space = env.action_space
obs_dim = obs_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model architecture
model_type = config.get("model_type", "basic_lstm").lower()
model = None
if algo == "dqn":
    # Use dueling transformer for DQN
    num_actions = act_space.n
    model = DuelingTransformerNet(obs_dim, num_actions)
else:
    #action_dim = act_space.shape[0] if isinstance(act_space, type(env.envs[0].action_space)) and hasattr(act_space, 'shape') else act_space.n
    action_dim = act_space.shape[0] if isinstance(act_space, gym.spaces.Box) else act_space.n
    if model_type == "basic_lstm":
        model = BasicLSTMNet(obs_dim, action_dim)
    elif model_type == "spatiotemporal_rln":
        model = SpatioTemporalRLNNet(obs_dim, action_dim)
    elif model_type == "transformer":
        model = TransformerNet(obs_dim, action_dim)
    elif model_type == "dueling_transformer":
        # For actor-critic, treat dueling as normal transformer
        model = TransformerNet(obs_dim, action_dim)
    else:
        model = BasicLSTMNet(obs_dim, action_dim)
# Load trained weights
model_file = model_path
# If DQN and no model_final, try q_net_final
if algo == "dqn" and not os.path.exists(model_path):
    model_file = f"{results_dir}/q_net_final.pth"
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

# Run evaluation episodes
num_eval_episodes = 5
total_rewards = []
for ep in range(num_eval_episodes):
    obs = env.reset()
    obs = np.array(obs)
    done = False
    ep_reward = 0.0
    # For RNN models, reset hidden state
    if isinstance(model, BasicLSTMNet) or isinstance(model, SpatioTemporalRLNNet):
        h_shape = (1, 1, model.hidden_dim)
        lstm_hidden = torch.zeros(h_shape).to(device)
        lstm_cell = torch.zeros(h_shape).to(device)
    while True:
        obs_tensor = torch.FloatTensor(obs).to(device)
        if algo == "dqn":
            # Greedy action from Q-network
            q_values = model(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).cpu().numpy()[0])
        else:
            if isinstance(model, BasicLSTMNet) or isinstance(model, SpatioTemporalRLNNet):
                # Single-step forward for LSTM
                if isinstance(model, SpatioTemporalRLNNet):
                    feat = model.feature_extractor(obs_tensor).unsqueeze(0)
                else:
                    feat = obs_tensor.unsqueeze(0)
                lstm_out, (lstm_hidden, lstm_cell) = model.lstm(feat, (lstm_hidden, lstm_cell))
                output = lstm_out.squeeze(0)
                mean_action = model.actor(output)
                # For continuous actions, take mean as deterministic action
                if isinstance(act_space, type(env.envs[0].action_space)) and hasattr(act_space, 'shape'):
                    action = mean_action.cpu().detach().numpy()[0]
                    # Clip to action space bounds
                    action = np.clip(action, act_space.low, act_space.high)
                else:
                    action = int(torch.argmax(mean_action, dim=1).cpu().numpy()[0])
            else:
                # Feed-forward model
                action_mean, _ = model(obs_tensor)
                if isinstance(act_space, type(env.envs[0].action_space)) and hasattr(act_space, 'shape'):
                    action = action_mean.cpu().detach().numpy()[0]
                    action = np.clip(action, act_space.low, act_space.high)
                else:
                    action = int(torch.argmax(action_mean, dim=1).cpu().numpy()[0])
        # Step environment
        obs, reward, done, info = env.step(action)
        obs = np.array(obs)
        ep_reward += reward[0]
        # Render environment if applicable (comment out if running headless)
        env.render(mode="human")
        if done[0]:
            print(f"Episode {ep+1}: Total Reward = {ep_reward}")
            total_rewards.append(ep_reward)
            break

# Print average reward over evaluation episodes
if total_rewards:
    avg_rew = np.mean(total_rewards)
    print(f"Average Reward over {num_eval_episodes} episodes: {avg_rew}")
