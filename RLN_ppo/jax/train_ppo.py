# train_ppo_jax.py
import os
import yaml
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
import optax
import distrax
from flax.training.train_state import TrainState

from f1env_jax import make_vectorized_env
from models_jax import BasicLSTMActorCritic, GaussianActor, Critic

# -----------------------------------------------------------------------------
# 1) Configuration container
# -----------------------------------------------------------------------------
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
    model_type: str

def compute_gae(
    rewards: jnp.ndarray,    # [T, N]
    values: jnp.ndarray,     # [T, N]
    last_value: jnp.ndarray, # [N]
    gamma: float,
    lam: float
) -> (jnp.ndarray, jnp.ndarray):
    T, N = rewards.shape
    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros((N,))
    values_ext = jnp.concatenate([values, last_value[None, :]], axis=0)  # [T+1, N]

    def scan_fn(carry, idx):
        gae = carry
        t = T - 1 - idx
        delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]
        gae = delta + gamma * lam * gae
        return gae, gae

    # scan backwards
    _, advs_back = jax.lax.scan(scan_fn, gae, jnp.arange(T))
    advantages = jnp.flip(advs_back, axis=0)
    returns = advantages + values
    return advantages, returns

# -----------------------------------------------------------------------------
# 3) Main training routine
# -----------------------------------------------------------------------------
def main():
    # a) load YAML
    with open("/Users/shyryn/Desktop/YC/rln/RL_RecurrentLidarNet/RL_RecurrentLidarNet/RLN_ppo/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = PPOConfig(
        seed=cfg["seed"],
        num_envs=cfg["n_envs"],
        rollout_steps=cfg["rollout_steps"],
        total_timesteps=cfg["total_timesteps"],
        ppo_epochs=cfg["ppo_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_eps=cfg["ppo_clip"],
        learning_rate=cfg["learning_rate"],
        ent_coef=cfg.get("ent_coef", 0.0),
        vf_coef=cfg.get("vf_coef", 0.5),
        hidden_dim=cfg["model"]["hidden_dim"],
        obs_dim=cfg["model"]["obs_dim"],
        action_dim=cfg["model"]["action_dim"],
        model_type=cfg["model"]["type"],
    )

    # b) make vectorized env
    reset_fn, step_fn, keys = make_vectorized_env(cfg, ppo_cfg.num_envs, ppo_cfg.seed)

    # c) initial reset
    obs_dict, state = reset_fn(keys)                  # obs_dict: { 'agent_0': [N, obs_dim] }
    obs = obs_dict["agent_0"]                        # [N, obs_dim]

    # d) build networks
    if ppo_cfg.model_type == "bilstm":
        net = BasicLSTMActorCritic(
            obs_dim=ppo_cfg.obs_dim,
            action_dim=ppo_cfg.action_dim,
            hidden_dim=ppo_cfg.hidden_dim,
            bidirectional=cfg["model"]["bidirectional"],
        )
        # carry will be threaded through time
        carry = None
        # initialize carry & params
        rng = jax.random.PRNGKey(ppo_cfg.seed)
        init_vars = net.init(rng, obs, carry)
        params = init_vars["params"]
    else:
        actor = GaussianActor(ppo_cfg.hidden_dim, ppo_cfg.action_dim)
        critic = Critic(ppo_cfg.hidden_dim)
        rng = jax.random.PRNGKey(ppo_cfg.seed)
        actor_params = actor.init(rng, obs)
        critic_params = critic.init(rng, obs)
        params = {"actor": actor_params, "critic": critic_params}

    # e) optimizer
    tx = optax.adam(ppo_cfg.learning_rate)
    train_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx
    )

    # f) training loop
    num_updates = ppo_cfg.total_timesteps // (ppo_cfg.rollout_steps * ppo_cfg.num_envs)
    for update in range(num_updates):
        # buffers: [T, N, ...]
        obs_buf     = []
        act_buf     = []
        logp_buf    = []
        rew_buf     = []
        val_buf     = []
        done_buf    = []

        for t in range(ppo_cfg.rollout_steps):
            # 1) forward pass
            if ppo_cfg.model_type == "bilstm":
                dist, value, carry = net.apply(
                    {"params": train_state.params}, obs, carry
                )
            else:
                dist = actor.apply({"params": train_state.params["actor"]}, obs)
                value = critic.apply({"params": train_state.params["critic"]}, obs)

            # 2) sample actions
            rng, act_key = jax.random.split(rng)
            actions = dist.sample(seed=act_key)         # [N, action_dim]
            logp    = dist.log_prob(actions).sum(-1)   # [N]

            # 3) step environment
            # split per-env keys for stepping
            split = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
            keys, step_keys = split[:, 1], split[:, 0]

            obs_next_dict, state, rewards, dones, _ = step_fn(
                step_keys, state, {"agent_0": actions}
            )
            obs_next = obs_next_dict["agent_0"]

            # 4) store
            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logp)
            rew_buf.append(rewards)
            val_buf.append(value)
            done_buf.append(dones["agent_0"])

            obs = obs_next

            # reset carry on done
            if ppo_cfg.model_type == "bilstm":
                # mask carry for finished episodes
                reset_mask = dones["agent_0"][:, None]
                carry = jax.tree_map(
                    lambda c: jnp.where(reset_mask, jnp.zeros_like(c), c),
                    carry
                )

        # stack buffers into arrays [T, N, ...]
        obs_arr   = jnp.stack(obs_buf)
        act_arr   = jnp.stack(act_buf)
        logp_arr  = jnp.stack(logp_buf)
        rew_arr   = jnp.stack(rew_buf)
        val_arr   = jnp.stack(val_buf)

        # last value for GAE
        if ppo_cfg.model_type == "bilstm":
            _, last_val, _ = net.apply({"params": train_state.params}, obs, carry)
        else:
            last_val = critic.apply({"params": train_state.params["critic"]}, obs)

        # GAE
        adv_arr, ret_arr = compute_gae(
            rew_arr, val_arr, last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda
        )

        # flatten all [T, N, ...] -> [T*N, ...]
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])

        b_obs   = flatten(obs_arr)
        b_act   = flatten(act_arr)
        b_logp  = logp_arr.reshape(-1)
        b_adv   = adv_arr.reshape(-1)
        b_ret   = ret_arr.reshape(-1)

        # PPO update
        def loss_fn(params, obs, act, old_logp, adv, ret):
            if ppo_cfg.model_type == "bilstm":
                pi, vpred, _ = net.apply({"params": params}, obs, carry=None)
            else:
                pi   = actor.apply({"params": params["actor"]}, obs)
                vpred = critic.apply({"params": params["critic"]}, obs)

            logp = pi.log_prob(act).sum(-1)
            ratio = jnp.exp(logp - old_logp)
            pg_loss = -jnp.mean(jnp.minimum(ratio * adv,
                jnp.clip(ratio, 1-ppo_cfg.clip_eps, 1+ppo_cfg.clip_eps) * adv))
            v_loss = jnp.mean((vpred - ret)**2)
            ent    = jnp.mean(pi.entropy())
            return pg_loss - ppo_cfg.ent_coef*ent + ppo_cfg.vf_coef*v_loss

        grads = jax.grad(loss_fn)(train_state.params,
                                 b_obs, b_act, b_logp, b_adv, b_ret)
        train_state = train_state.apply_gradients(grads=grads)

        if update % 10 == 0:
            print(f"[{update}/{num_updates}]")

if __name__ == "__main__":
    main()
