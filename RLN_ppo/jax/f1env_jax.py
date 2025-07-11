"""
JAX wrapper for F1TENTH-Gym (F110Env) with Frenet reward shaping
"""
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Tuple, Dict
import chex

from f110_env import F110Env
from utils import State, Param

class F110EnvJaxSingle:
    def __init__(self, config: Dict, seed: int = 0):
        self.params = Param(**config)
        # single-agent env
        self.env = F110Env(num_agents=1, params=self.params)
        # PRNG key -? why
        self.rng = random.PRNGKey(seed)

        self.last_lin_vel = 0.0
        self.last_str_ang = 0.0
        self.last_frenet_s = None
        self.timestep = config.get('timestep', 0.1)

        # JIT compile core methods
        self._reset_jit = jax.jit(self.env.reset)
        self._step_jit  = jax.jit(self.env.step_env)

    def reset(self) -> Tuple[jnp.ndarray, State]:
        self.rng, key = random.split(self.rng)
        obs_dict, state = self._reset_jit(key)
        obs = obs_dict['agent_0']
        s, _, _ = state.frenet_states[0]
        self.last_frenet_s = float(s)
        self.last_lin_vel = float(state.cartesian_states[0, 3])
        self.last_str_ang = float(state.cartesian_states[0, 4])
        return obs, state

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, State, float, bool, Dict]:
        self.rng, key = random.split(self.rng)
        obs_dict, state, _, dones, info = self._step_jit(
            key,
            state := state,
            {'agent_0': action}
        )
        obs = obs_dict['agent_0']

        s, ey, _ = state.frenet_states[0]
        lin_v = state.cartesian_states[0, 3]
        ang_v = state.cartesian_states[0, 4]
        coll  = state.collisions[0]

        ds = jnp.where(
            (self.last_frenet_s is None) | (s - self.last_frenet_s < -10.0),
            0.0,
            s - self.last_frenet_s
        )
        progress_rew = ds
        speed_rew    = jnp.abs(lin_v) - 1.0
        collision_pen= jnp.where(coll, -1000.0, 0.0)
        steer_rate   = jnp.abs(action[0] - self.last_str_ang) / self.timestep
        steer_pen    = -0.05 * steer_rate

        reward = (
              20.0 * progress_rew
            +  1.0 * speed_rew
            +        collision_pen
            +        steer_pen
        )

        self.last_frenet_s = float(s)
        self.last_lin_vel   = float(lin_v)
        self.last_str_angf  = float(action[0])

        done = dones['agent_0']
        return obs, state, reward, done, info

def make_vectorized_env(config: Dict, num_envs: int, seed: int = 0):
    wrapper = F110EnvJaxSingle(config, seed)
    keys = random.split(wrapper.rng, num_envs)
    reset_fn = jax.vmap(lambda k: wrapper._reset_jit(k), in_axes=0)
    step_fn  = jax.vmap(lambda k, st, a: wrapper._step_jit(k, st, {'agent_0': a}), in_axes=(0,0,0))
    return reset_fn, step_fn, keys
