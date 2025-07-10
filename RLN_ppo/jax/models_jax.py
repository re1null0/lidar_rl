import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import distrax


class GaussianActor(nn.Module):
    """Feedforward Gaussian policy for continuous action spaces."""
    hidden_dim: int
    action_dim: int
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> distrax.Distribution:
        # choose activation
        act = nn.tanh if self.activation == 'tanh' else nn.relu
        # two hidden layers
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
        x = act(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
        x = act(x)
        # output mean
        mu = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        # learnable log-std parameter
        log_std = self.param('log_std', constant(0.0), (self.action_dim,))
        std = jnp.exp(log_std)
        # return a normal distribution
        return distrax.Normal(loc=mu, scale=std)


class Critic(nn.Module):
    """Feedforward state-value estimator."""
    hidden_dim: int
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = nn.tanh if self.activation == 'tanh' else nn.relu
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
        x = act(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal())(x)
        x = act(x)
        v = nn.Dense(1, kernel_init=orthogonal(0.01))(x)
        return jnp.squeeze(v, axis=-1)
