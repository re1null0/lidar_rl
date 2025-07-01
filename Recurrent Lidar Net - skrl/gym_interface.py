import os
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from skrl.envs.wrappers.torch import wrap_env

# Custom environment wrapper for F1TENTH gym to handle observation processing, noise, and reward shaping.
class F110EnvWrapper(gym.Env):
    def __init__(self, config, seed=0):
        super().__init__()
        self.config = config
        self.seed = seed
        # Apply domain randomization to dynamics if enabled
        params = None
        if config.get("domain_randomization", False):
            rng = np.random.RandomState(seed)
            params = {
                'mu': rng.uniform(0.8, 1.2),
                'C_Sf': rng.uniform(4.0, 5.5),
                'C_Sr': rng.uniform(4.0, 5.5),
                'm': rng.uniform(3.0, 4.5),
                'I': rng.uniform(0.04, 0.05)
            }
        
        # Create the underlying F1TENTH gym environment
        env_id = config.get("env_id", "f1tenth_gym:f1tenth-v0")
        map_path = config.get("map_path", None)
        self.env = gym.make(env_id, seed=seed, map=map_path, params=params, model='dynamic_ST', num_agents=1)

        # If max_episode_steps is set, (optional) handle termination after that many steps
        self._max_episode_steps = config.get("max_episode_steps", None)
        self.current_step = 0

        # Determine observation space dimensions
        lidar_enabled = config["lidar"]["enabled"]
        lidar_downsample = config["lidar"]["downsample"]
        full_dim = 1080
        lidar_dim = 108 if (lidar_enabled and lidar_downsample) else (full_dim if lidar_enabled else 0)
        state_dim = 1 if config.get("include_velocity_in_obs", True) else 0
        obs_dim = state_dim + lidar_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Determine action space: continuous for PPO/A2C, discrete for DQN
        self.discrete_actions = None
        if config["algorithm"].lower() == "dqn":
            # Define a discrete action set (steering, velocity pairs) for DQN
            self.discrete_actions = np.array([
                [-0.4, 5.0],   # steer left, medium speed
                [ 0.0, 5.0],   # go straight, medium speed
                [ 0.4, 5.0],   # steer right, medium speed
                [ 0.0, 2.0],   # go straight, slow speed
                [ 0.0, 8.0]    # go straight, fast speed
            ], dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        else:
            # Continuous action space (steering, velocity)
            self.action_space = self.env.action_space

        # Sensor noise parameters
        noise_cfg = config.get("sensor_noise", {})
        self.lidar_noise_std = noise_cfg.get("lidar", 0.0)
        self.speed_noise_std = noise_cfg.get("speed", 0.0)

        # Variables for tracking last values and cumulative metrics
        self.last_speed = 0.0
        self.last_steer = 0.0
        self.total_abs_speed = 0.0
        self.total_abs_steer_change = 0.0

    def _extract_speed(self, obs):
        """Extract forward speed from environment observation dict (handles different keys)."""
        try:
            return obs['agent_0']['std_state'][3]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Observation dict does not contain expected keys for speed extraction.")
        



        

    def _extract_lidar(self, obs):
        """Extract LiDAR scan array from observation dict (handles different keys)."""
        for k in ("scans", "lidar", "laser_scan", "ranges"):
            if k in obs:
                return obs[k][0] if hasattr(obs[k], "__len__") else obs[k]
        return None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment and return initial observation."""
        self.current_step = 0
        if seed is not None:
            self.seed = seed
        # Forward the reset call to the underlying env
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        result = self.env.reset(**kwargs)
        obs_dict = result[0] if isinstance(result, tuple) else result

        # Reset tracking variables
        self.last_speed = 0.0
        self.last_steer = 0.0
        self.total_abs_speed = 0.0
        self.total_abs_steer_change = 0.0

        processed_obs = self._process_obs(obs_dict)
        info = {}
        return processed_obs, info

    def step(self, action):
        """Step the environment with the given action and return processed observation, reward, done flags, and info."""
        self.current_step += 1
        # Map discrete action index to actual action values if using discrete actions
        if self.discrete_actions is not None:
            actual_action = self.discrete_actions[action]
        else:
            actual_action = np.array(action, dtype=np.float32)
            if actual_action.ndim == 1:
                actual_action = actual_action[None, :]  # add batch dimension for single agent

        # Ensure action shape matches number of agents in underlying env
        try:
            n_agents = len(self.env.agents)  # for newer f1tenth_gym versions
        except AttributeError:
            n_agents = getattr(self.env, "num_agents", 1)
        if actual_action.ndim == 1:
            actual_action = np.tile(actual_action, (n_agents, 1))
        elif actual_action.shape[0] != n_agents:
            actual_action = np.tile(actual_action[0], (n_agents, 1))

        # Step the underlying environment
        result = self.env.step(actual_action)
        if len(result) == 5:
            obs_dict, env_reward, terminated, truncated, info = result
        else:
            obs_dict, env_reward, done, info = result
            terminated, truncated = done, False  # ensure terminated/truncated flags

        # Compute shaped reward components
        speed = abs(self._extract_speed(obs_dict))
        steer = float(actual_action[0, 0])  # steering command of first agent
        steering_delta = abs(steer - self.last_steer)
        progress_reward = 0.0  # TODO: (optional track progress if available)
        speed_reward = speed - 1.0  # reward for speed (baseline 1 m/s)
        accel = abs(speed - self.last_speed)
        collision = bool(obs_dict.get("collisions", [False])[0])
        # Combine reward with weights
        weight = self.config["reward_weights"]
        reward = weight.get("progress", 0.0) * progress_reward \
               + weight.get("speed", 0.0) * speed_reward \
               - weight.get("steering_change", 0.0) * steering_delta \
               - weight.get("acceleration", 0.0) * accel \
               - (weight.get("collision", 0.0) * 1.0 if collision else 0.0)

        # Update cumulative metrics and last values
        self.total_abs_speed += speed
        self.total_abs_steer_change += steering_delta
        self.last_speed = speed
        self.last_steer = steer

        processed_obs = self._process_obs(obs_dict)
        # If episode ends, add custom metrics to info for logging
        if terminated or truncated:
            info_episode = info.get("episode", {})
            info_episode["avg_speed"] = (self.total_abs_speed / self.current_step) if self.current_step > 0 else 0.0
            info_episode["avg_steering_change"] = (self.total_abs_steer_change / self.current_step) if self.current_step > 0 else 0.0
            info["episode"] = info_episode

        return processed_obs, reward, terminated, truncated, info


    '''
    def _process_obs(self, obs_dict):
        """Return a flat vector of length self.observation_space.shape[0]."""
        target_len = self.observation_space.shape[0]
        vec        = np.zeros(target_len, dtype=np.float32)   # pre-allocate

        idx = 0
        # ───── speed ───────────────────────────────────────────────
        if self.config.get("include_velocity_in_obs", True):
            speed = self._extract_speed(obs_dict)
            if self.speed_noise_std > 0:
                speed += np.random.normal(0, self.speed_noise_std)
            vec[idx] = speed
            idx += 1

        # ───── LiDAR ───────────────────────────────────────────────
        if self.config["lidar"]["enabled"]:
            scan = self._extract_lidar(obs_dict)
            if scan is not None:
                if self.config["lidar"]["downsample"]:
                    scan = scan[::10]                        # 108 readings if 1080 raw
                if self.lidar_noise_std > 0:
                    scan = np.clip(
                        scan + np.random.normal(0, self.lidar_noise_std, size=scan.shape) * scan,
                        0.0, np.inf
                    )
                scan_len          = min(len(scan), target_len - idx)
                vec[idx: idx+scan_len] = scan[:scan_len]

        return vec
    '''

    
    def _process_obs(self, obs_dict):
        #Process raw observation dict into a flat NumPy array (add noise, downsample LiDAR, etc.).
        obs_vec = []

        # Include velocity (speed) in observation if enabled
        if self.config.get("include_velocity_in_obs", True):
            speed = self._extract_speed(obs_dict)
            if self.speed_noise_std > 0:
                speed += np.random.normal(0, self.speed_noise_std)
            obs_vec.append(speed)

        # Include LiDAR scan if enabled
        lidar_scan = None
        if self.config["lidar"]["enabled"]:
            lidar_scan = self._extract_lidar(obs_dict)
        if lidar_scan is not None:
            # Downsample LiDAR if configured
            if self.config["lidar"]["downsample"]:
                lidar_scan = lidar_scan[::10]
            # Add noise to LiDAR readings (proportional noise)
            if self.lidar_noise_std > 0:
                noise = np.random.normal(0, self.lidar_noise_std, size=lidar_scan.shape)
                lidar_scan = np.clip(lidar_scan + noise * lidar_scan, 0.0, np.inf)
            obs_vec.extend(lidar_scan.astype(np.float32))
        obs_array = np.array(obs_vec, dtype=np.float32)

        ###### SAFETY PAD so len(obs_vec) == self.observation_space.shape[0] ####
        target = self.observation_space.shape[0]
        if len(obs_vec) < target:
            obs_vec.extend([0.0] * (target - len(obs_vec)))      # pad missing slots
        elif len(obs_vec) > target:
            obs_vec = obs_vec[:target]                           # rare: trim extras

        obs_array = np.array(obs_vec, dtype=np.float32)
        return obs_array
    

def make_vector_env(config):
    """Create a vectorized environment with the specified number of parallel F1TENTH environments."""
    num_envs = config.get("n_envs", 1)
    use_subproc = (num_envs > 1 and os.name != 'nt')  # use subprocesses for parallel envs if not on Windows
    def make_env_fn(rank):
        def _init():
            env = F110EnvWrapper(config, seed=config.get("seed", 0) + rank)
            # Wrap each env to record episode statistics (returns and lengths)
            return RecordEpisodeStatistics(env)
        return _init
    if num_envs == 1:
        env = F110EnvWrapper(config, seed=config.get("seed", 0))
        #env = RecordEpisodeStatistics(env)
        env = wrap_env(env)  # wrap single environment for skrl
    else:
        env_fns = [make_env_fn(i) for i in range(num_envs)]
        env = AsyncVectorEnv(env_fns) if use_subproc else SyncVectorEnv(env_fns)
        env = wrap_env(env)
    return env