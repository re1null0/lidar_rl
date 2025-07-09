import os
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from track import Track
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
        
        env_id   = config.get("env_id", "f1tenth_gym:f1enth-v0")
        map_path = config.get("map_path", None)
        # Track() will read the same YAML, build centerline & .waypoints
        self.track     = Track(map_path)
        waypoints      = self.track.waypoints
        timestep       = config.get("timestep", 0.1)
        self.env = gym.make(
            env_id,
            seed=seed,
            map=map_path,
            params=params,
            model="dynamic_ST",
            num_agents=1,
            waypoints=waypoints,
            timestep=timestep
        )

        # If max_episode_steps is set, (optional) handle termination after that many steps
        self._max_episode_steps = config.get("max_episode_steps", None)
        self.current_step = 0
        self.last_linear_velocity    = None
        self.last_steering_angle     = None
        self.last_frenet_arc_length  = None
        self.timestep                = timestep
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
        

    def cartesian_to_frenet(self, x, y, phi, s_guess=None):
        if s_guess is None:
            s_guess = self.s_guess

        s, ey = self.centerline.calc_arclength(x, y, s_guess)
        s = s % self.s_frame_max
        self.s_guess = s

        yaw = self.centerline.calc_yaw(s)
        normal = np.asarray([-np.sin(yaw), np.cos(yaw)])
        x_eval, y_eval = self.centerline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign
        phi = phi - yaw
        return s, ey, np.arctan2(np.sin(phi), np.cos(phi))


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
        self.last_frenet_s = None
        self.total_abs_speed = 0.0
        self.total_abs_steer_change = 0.0

        processed_obs = self._process_obs(obs_dict)
        info = {}
        return processed_obs, info

    def step(self, action):
        self.current_step += 1

        if self.discrete_actions is not None:
            actual_action = self.discrete_actions[action]
        else:
            actual_action = np.array(action, dtype=np.float32)
            if actual_action.ndim == 1:
                actual_action = actual_action[None, :]

        try:
            n_agents = len(self.env.get_wrapper_attr('agents')) \
                if self.env.get_wrapper_attr('agents') is not None \
                else getattr(self.env.unwrapped, "num_agents", 1)
        except AttributeError:
            n_agents = getattr(self.env, "num_agents", 1)
        if actual_action.ndim == 1:
            actual_action = np.tile(actual_action, (n_agents, 1))
        elif actual_action.shape[0] != n_agents:
            actual_action = np.tile(actual_action[0], (n_agents, 1))

        result = self.env.step(actual_action)
        if len(result) == 5:
            obs_dict, env_reward, terminated, truncated, info = result
        else:
            obs_dict, env_reward, done, info = result
            terminated, truncated = done, False

        #rl_env.py logic
        linear_velocity       = obs_dict["linear_vels_x"][0]
        frenet_arc_length     = obs_dict["state_frenet"][0][0]
        frenet_lateral_offset = obs_dict["state_frenet"][0][1]
        collision             = bool(obs_dict["collisions"][0])
        angular_velocity      = obs_dict["ang_vels_z"][0]

        linear_acceleration  = 0.0
        steering_angle_speed = 0.0
        if self.last_linear_velocity is not None:
            linear_acceleration = abs(linear_velocity - self.last_linear_velocity) / self.timestep
        if self.last_steering_angle is not None:
            steering_angle_speed = abs(float(actual_action[0,0]) - self.last_steering_angle) / self.timestep

        if (self.last_frenet_arc_length is not None and
            frenet_arc_length - self.last_frenet_arc_length < -10):
            self.last_frenet_arc_length = None

        if self.last_frenet_arc_length is None:
            progress_reward = 0.0
        else:
            progress_reward = frenet_arc_length - self.last_frenet_arc_length
        self.last_frenet_arc_length = frenet_arc_length

        safety_distance_reward     = 0.5 - abs(frenet_lateral_offset)
        linear_velocity_reward     = abs(linear_velocity) - 1.0
        collision_punishment       = -1.0 if collision else 0.0
        angular_velocity_punishment= -abs(angular_velocity)
        acceleration_punishment    = -linear_acceleration
        steering_speed_punishment  = -steering_angle_speed

        # single-agent weights
        reward = (
              20.0   * progress_reward
            +  0.0   * safety_distance_reward
            +  1.0   * linear_velocity_reward
            +1000.0  * collision_punishment
            +  0.0   * angular_velocity_punishment
            +  0.0   * acceleration_punishment
            +  0.05  * steering_speed_punishment
        )

        self.last_linear_velocity   = linear_velocity
        self.last_steering_angle    = float(actual_action[0,0])

        processed_obs = self._process_obs(obs_dict)
        return processed_obs, reward, terminated, truncated, info

    
    def _process_obs(self, obs_dict):
        #Process raw observation dict into a flat NumPy array (add noise, downsample LiDAR, etc.).
        obs_vec = []

        # Include velocity (speed) in observation if enabled
        if self.config.get("include_velocity_in_obs", True):
            speed = self._extract_speed(obs_dict)
            if self.speed_noise_std > 0:
                speed += np.random.normal(0, self.speed_noise_std)
            obs_vec.append(speed)

        s_raw, d_raw = obs_dict["state_frenet"][0]

        # 2b) Clip to sane ranges (just like rl_env does)
        s = float(np.clip(s_raw, -1000.0, 1000.0))
        d = float(np.clip(d_raw,   -5.0,    5.0))

        # 2c) Add them at the front of your observation vector
        obs_vec.extend([s, d])

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
        print('use_subproc', use_subproc)
        # env_fns = [make_env_fn(i) for i in range(num_envs)]
        def make_env():
            env = F110EnvWrapper(config, seed=config.get("seed", 0))
            return env
        # env = AsyncVectorEnv(env_fns) if use_subproc else SyncVectorEnv(env_fns)
        env = gym.vector.AsyncVectorEnv([make_env for _ in range(config['n_envs'])])
        env = wrap_env(env)
        
    return env