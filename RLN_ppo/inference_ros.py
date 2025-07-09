#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import yaml
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from models import BasicLSTMNet, SpatioTemporalRLNNet, TransformerNet, DuelingTransformerNet, SpatioTempDuelingTransformerNet
from train_ppo import PolicyWrapper
import os

class InferenceROS(Node):
    def __init__(self, model, config, device):
        super().__init__('inference_ros_node')

        # ROS subscriptions and publications
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        # Model and configuration
        self.model = model
        self.device = device
        self.config = config
        self.obs_dim = config['obs_dim']
        self.lidar_downsample = config['lidar']['downsample']
        self.include_vel = config['include_velocity_in_obs']
        self.drive_msg = AckermannDriveStamped()

        # LSTM hidden states (if applicable)
        self.lstm_hidden = None
        self.lstm_cell = None
        if isinstance(model.base, BasicLSTMNet) or isinstance(model.base, SpatioTemporalRLNNet):
            h_dim = model.base.hidden_dim if hasattr(model.base, "hidden_dim") else 128
            self.lstm_hidden = torch.zeros((1, 1, h_dim)).to(device)
            self.lstm_cell = torch.zeros((1, 1, h_dim)).to(device)

        self.get_logger().info("InferenceROS Node Initialized.")

    def preprocess_lidar(self, ranges):
        return ranges  # No downsampling

    def lidar_callback(self, data: LaserScan):
        # Extract LiDAR scan
        lidar_scan = self.preprocess_lidar(data.ranges)

        # Create observation vector
        obs_vec = []
        if self.include_vel:
            # Placeholder for velocity (can be replaced with actual velocity data if available)
            obs_vec.append(0.0)
        obs_vec.extend(lidar_scan.tolist())

        # Pad or trim observation vector to match obs_dim
        if len(obs_vec) < self.obs_dim:
            obs_vec.extend([0.0] * (self.obs_dim - len(obs_vec)))
        elif len(obs_vec) > self.obs_dim:
            obs_vec = obs_vec[:self.obs_dim]

        # Convert observation to tensor
        obs_array = np.array(obs_vec, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_array).to(self.device).unsqueeze(0)

        # Perform inference
        if isinstance(self.model.base, BasicLSTMNet) or isinstance(self.model.base, SpatioTemporalRLNNet):
            inp = obs_tensor.unsqueeze(0)  # Shape: (1, 1, obs_dim)
            action_mean, _ = self.model.base(inp)
        else:
            action_mean = self.model.base(obs_tensor)

        # Extract action
        action = action_mean.detach().cpu().numpy().flatten()
        steering_angle = float(action[0])  # Explicitly cast to float
        velocity = float(action[1])        # Explicitly cast to float


        # steering_angle, 
        velocity = 3.0

        # Publish drive message
        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)

        self.get_logger().info(f"Steering={steering_angle:.2f} rad, Speed={velocity:.1f}")

def main(args=None):
    rclpy.init(args=args)

    # Load configuration and model
    base_results_dir = "/home/saichand/ros2_ws/src/RL_RecurrentLidarNet/RLN_ppo/results"
    results_dir = max(
        [os.path.join(base_results_dir, d) for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d))],
        key=os.path.getmtime
    )

    config_path = f"{results_dir}/config.yaml"
    model_path = f"{results_dir}/f1tenth_experiment/checkpoints/best_agent.pt"

    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)

    # Get model configuration
    model_type = config.get("model_type", "spatiotemporal_rln").lower()
    obs_dim = config.get("obs_dim", 1081)  # Use obs_dim from config
    action_dim = 2  # steering and velocity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create observation/action spaces (needed for PolicyWrapper)
    from gymnasium.spaces import Box
    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    action_space = Box(low=-np.inf, high=np.inf, shape=(action_dim,))

    # Initialize the correct model type from config
    base_model = None
    if model_type == "basic_lstm":
        base_model = BasicLSTMNet(obs_dim, action_dim)
    elif model_type == "spatiotemporal_rln":
        base_model = SpatioTemporalRLNNet(obs_dim, action_dim)
    elif model_type == "transformer":
        base_model = TransformerNet(obs_dim, action_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    base_model = base_model.to(device)
    
    # Wrap the base model
    model = PolicyWrapper(base_model, obs_space, action_space, device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "policy" in checkpoint:
        model.load_state_dict(checkpoint["policy"])
    else:
        raise RuntimeError("Checkpoint does not contain 'policy' weights.")

    model.eval()

    # Update config with correct dimensions
    config['obs_dim'] = obs_dim
    config['action_dim'] = action_dim

    # Start ROS node
    node = InferenceROS(model, config, device)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()