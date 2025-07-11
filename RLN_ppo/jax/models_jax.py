import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import distrax
from typing import Any, Tuple

class BasicLSTMActorCritic(nn.Module):
    """
    One‐layer (bi‐)LSTM + separate actor & critic heads
    Input x is either [batch, obs_dim] or [seq_len, batch, obs_dim].
    Returns a distrax.Normal policy, a value scalar, and new LSTM carry.
    """
    obs_dim: int
    action_dim: int
    hidden_dim: int = 128
    bidirectional: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        carry: Tuple[Any, Any] = None,    # (hidden, cell) for each direction
    ) -> Tuple[distrax.Distribution, jnp.ndarray, Tuple[Any,Any]]:
        # ensure time dim
        if x.ndim == 2:
            # [batch, obs_dim] -> [1, batch, obs_dim]
            x = x[jnp.newaxis, ...]

        # build LSTM
        lstm = nn.LSTM(
            hidden_size=self.hidden_dim,
            bidirectional=self.bidirectional,
            name="lstm",
        )

        # initialize carry if not provided
        batch_size = x.shape[1]
        if carry is None:
            carry = lstm.initialize_carry(
                jax.random.PRNGKey(0),  # you should pass in a real key if you need determinism
                (batch_size,),
            )

        # run the LSTM over the sequence
        outputs, new_carry = lstm(carry, x)  
        # outputs: [seq_len, batch, hidden_dim*(2 if bi else 1)]

        # grab last time‐step
        feat = outputs[-1]  # [batch, hidden_dim * (2 if bidir else 1)]

        # actor head: mean
        actor_mean = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc1"
        )(feat)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_fc2"
        )(actor_mean)

        # learnable log‐std
        log_std = self.param("log_std", constant(0.0), (self.action_dim,))
        std = jnp.exp(log_std)

        pi = distrax.Normal(actor_mean, std)

        # critic head
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc1"
        )(feat)
        critic = nn.tanh(critic)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="critic_fc2"
        )(critic)
        value = value.squeeze(-1)  # [batch]

        return pi, value, new_carry

# import jax.numpy as jnp
# import flax.linen as nn
# from flax.linen.initializers import orthogonal, constant
# import distrax


# class GaussianActor(nn.Module):
#     """Feedforward Gaussian policy for continuous action spaces."""
#     hidden_dim: int
#     action_dim: int
#     activation: str = 'tanh'

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> distrax.Distribution:
#         # choose activation
#         act = nn.tanh if self.activation == 'tanh' else nn.relu
#         # two hidden layers
#         x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
#         x = act(x)
#         x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
#         x = act(x)
#         # output mean
#         mu = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
#         # learnable log-std parameter
#         log_std = self.param('log_std', constant(0.0), (self.action_dim,))
#         std = jnp.exp(log_std)
#         # return a normal distribution
#         return distrax.Normal(loc=mu, scale=std)


# class Critic(nn.Module):
#     """Feedforward state-value estimator."""
#     hidden_dim: int
#     activation: str = 'tanh'

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         act = nn.tanh if self.activation == 'tanh' else nn.relu
#         x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
#         x = act(x)
#         x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
#         x = act(x)
#         v = nn.Dense(1, kernel_init=orthogonal(0.01))(x)
#         return jnp.squeeze(v, axis=-1)
