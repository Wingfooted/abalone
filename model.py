from flax import linen as nn
from typing import Sequence, Callable


class Model(nn.Module):
    model_layout: Sequence[int] = (4, 5, 5, 4)
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, x):
        for layer_width in self.model_layout:
            x = nn.Dense(layer_width,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return nn.relu(x)
