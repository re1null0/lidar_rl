import os
import yaml
import jax
import jax.numpy as jnp
import optax
import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict, Any, Tuple
from f1env_jax import make_vectorized_env

# PPO hyperparameter container
class PPOConfig(NamedTuple):
    seed: int
    num_envs: int
    rollout_steps: int
    total_timesteps: int
    ppo_epochs: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    learning_rate: float
    ent_coef: float
    vf_coef: float
    hidden_dim: int
    obs_dim: int
    action_dim: int

# Simple MLP actor-critic
class GaussianActor(nn.Module):
    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        mu = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', lambda k: jnp.zeros((self.action_dim,)))
        std = jnp.exp(log_std)
        return distrax.Normal(mu, std)

class Critic(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        v = nn.Dense(1)(x)
        return jnp.squeeze(v, -1)

# GAE computation
def compute_gae(rewards, values, gamma, lam, last_value):
    adv = jnp.zeros_like(rewards)
    gae = 0.0
    values_ext = jnp.append(values, last_value)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]
        gae = delta + gamma * lam * gae
        adv = adv.at[t].set(gae)
    returns = adv + values
    return adv, returns

# Rollout collection
@jax.jit
def rollout_step(step_fn, keys, states, actions):
    next_obs, next_states, rewards, dones, infos = step_fn(keys, states, actions)
    return next_obs, next_states, rewards, dones, infos


def main():
    # load YAML config
    with open("config/default.yaml") as f:
        cfg_dict = yaml.safe_load(f)

    # build PPOConfig
    ppo_cfg = PPOConfig(
        seed=cfg_dict['seed'],
        num_envs=cfg_dict['n_envs'],
        rollout_steps=cfg_dict['rollout_steps'],
        total_timesteps=cfg_dict['total_timesteps'],
        ppo_epochs=cfg_dict['ppo_epochs'],
        gamma=cfg_dict['gamma'],
        gae_lambda=cfg_dict['gae_lambda'],
        clip_eps=cfg_dict['ppo_clip'],
        learning_rate=cfg_dict['learning_rate'],
        ent_coef=0.0,
        vf_coef=0.5,
        hidden_dim=cfg_dict['model']['hidden_dim'],
        obs_dim=cfg_dict['model']['obs_dim'],
        action_dim=cfg_dict['model']['action_dim']
    )

    # create vectorized env
    reset_fn, step_fn, keys = make_vectorized_env(cfg_dict, ppo_cfg.num_envs, ppo_cfg.seed)
    # initialize
    obs_batch, state_batch = reset_fn(keys)
    obs_batch = obs_batch['agent_0']  # shape (num_envs, obs_dim)

    # init networks
    actor = GaussianActor(ppo_cfg.hidden_dim, ppo_cfg.action_dim)
    critic = Critic(ppo_cfg.hidden_dim)
    rng = jax.random.PRNGKey(ppo_cfg.seed)
    actor_params = actor.init(rng, obs_batch)
    critic_params = critic.init(rng, obs_batch)

    tx = optax.adam(ppo_cfg.learning_rate)
    train_state = TrainState.create(
        apply_fn=None,
        params={'actor': actor_params, 'critic': critic_params},
        tx=tx
    )

    num_updates = ppo_cfg.total_timesteps // (ppo_cfg.rollout_steps * ppo_cfg.num_envs)
    for update in range(num_updates):
        # buffers
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        values_buf = []
        logp_buf = []
        dones_buf = []

        # rollout
        for t in range(ppo_cfg.rollout_steps):
            # policy & value
            dist = actor.apply({'params': train_state.params['actor']}, obs_batch)
            value = critic.apply({'params': train_state.params['critic']}, obs_batch)
            rng, act_key = jax.random.split(rng)
            action = dist.sample(seed=act_key)
            logp = dist.log_prob(action).sum(-1)

            # step env
            keys, step_keys = jax.random.split(keys, 2)
            obs_next_dict, state_batch, rewards, dones, infos = step_fn(
                step_keys, state_batch, {'agent_0': action}
            )
            obs_next = obs_next_dict['agent_0']

            # store
            obs_buf.append(obs_batch)
            actions_buf.append(action)
            rewards_buf.append(rewards)
            values_buf.append(value)
            logp_buf.append(logp)
            dones_buf.append(dones['agent_0'])

            obs_batch = obs_next

        # convert buffers
        obs_arr = jnp.stack(obs_buf)
        act_arr = jnp.stack(actions_buf)
        rew_arr = jnp.stack(rewards_buf)
        val_arr = jnp.stack(values_buf)
        logp_arr = jnp.stack(logp_buf)
        done_arr = jnp.stack(dones_buf)

        # last value for GAE
        last_value = critic.apply({'params': train_state.params['critic']}, obs_batch)
        adv, returns = compute_gae(rew_arr, val_arr, ppo_cfg.gamma, ppo_cfg.gae_lambda, last_value)

        # flatten
        b_obs = obs_arr.reshape(-1, ppo_cfg.obs_dim)
        b_act = act_arr.reshape(-1, ppo_cfg.action_dim)
        b_logp = logp_arr.reshape(-1)
        b_ret = returns.reshape(-1)
        b_adv = adv.reshape(-1)

        # PPO update
        def loss_fn(params, obs, act, old_logp, ret, adv):
            pi = actor.apply({'params': params['actor']}, obs)
            logp = pi.log_prob(act).sum(-1)
            ratio = jnp.exp(logp - old_logp)
            pg_loss = -jnp.mean(jnp.minimum(ratio * adv,
                jnp.clip(ratio, 1 - ppo_cfg.clip_eps, 1 + ppo_cfg.clip_eps) * adv))
            v = critic.apply({'params': params['critic']}, obs)
            v_loss = jnp.mean((v - ret)**2)
            ent = jnp.mean(pi.entropy())
            return pg_loss - ppo_cfg.ent_coef * ent + ppo_cfg.vf_coef * v_loss

        grads = jax.grad(loss_fn)(train_state.params, b_obs, b_act, b_logp, b_ret, b_adv)
        train_state = train_state.apply_gradients(grads=grads)

        if update % 10 == 0:
            print(f"Update {update}/{num_updates}")

if __name__ == '__main__':
    main()
