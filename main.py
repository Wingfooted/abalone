import jax
import jax.random as random
import jax.numpy as jnp
import pandas as pd

from flax import serialization
from model import Model

data = pd.read_csv("data/abalone.data", header=None)
column_names = [
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Viscera weight',
    'Shell weight',
    'Rings'
]

data.columns = column_names
data["M"] = data["Sex"].apply(lambda x: 1 if x == "M" else 0)
data["F"] = data["Sex"].apply(lambda x: 1 if x == "F" else 0)
data["I"] = data["Sex"].apply(lambda x: 1 if x == "I" else 0)
data = data.drop(columns=["Sex"])

rings = data.pop("Rings")
data['Rings'] = rings

train_df, test_df = data[:3800], data[3800:]


model_structure = (10, 10, 10, 10)
# model_rng, x_rng  = random.key(1), random.key(2)
# x = random.normal(x_rng, (10, ))

model = Model(model_layout=model_structure)

model_rng, x_rng = random.split(random.key(0))
x_sample = jnp.array(train_df.sample(1).iloc[:, :10].values[0])
params = model.init(model_rng, x_sample)
G = jax.tree.map(lambda p: jnp.zeros_like(p), params)  # for adagrad
# print(x_sample, params)
# params = model.init(model_rng, x)


def make_mse_loss(xs, ys):
    def mse_loss(params):
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y-pred, y-pred) / 2.0

        return jnp.mean(jax.vmap(squared_error)(xs, ys), axis=0)

    return jax.jit(mse_loss)


epoch_log_period = 10
epochs = 10000
lr = 0.005
clip = 5
epsilon = 1e-6


def get_data(train_df, sample_size=10, replace=True):
    df = train_df.sample(sample_size)
    x_df = df.iloc[:, :10]
    y_df = df.iloc[:, -1]
    xs = jnp.array([jnp.array(a) for a in x_df.values])
    ys = jnp.array([jnp.array(a) for a in y_df.values])
    return xs, ys


for epoch in range(epochs):
    xs, ys = get_data(train_df, sample_size=200)

    mse_loss = make_mse_loss(xs, ys)
    value_and_grad_fn = jax.value_and_grad(mse_loss)

    loss, grads = value_and_grad_fn(params)

    G = jax.tree.map(lambda G, g: G + jnp.square(g), G, grads)  # G is accumulated G, grads is grads from above
    #grads = jax.tree.map(lambda leaf: jnp.clip(leaf, -clip, clip) if isinstance(leaf, jax.Array) else leaf, params)
    params = jax.tree.map(lambda p, g, accum_g: p - (lr / jnp.sqrt(epsilon + accum_g) ) * g, params, grads, G)

    if epoch % epoch_log_period == 0:
        print(G)
        print(f"epoch: {epoch}, loss: {loss}")


# testing the model

# testing the model
residuals = []
for index, row in test_df.iterrows():
    row = row.values
    x, y = row[:10], row[-1]
    x, y = jnp.array(x), jnp.array(y)

    dist = model.apply(params, x) - y
    residuals.append(dist)

residuals_vector = jnp.array(residuals)
print(jnp.sum(residuals_vector))
print(jnp.mean(residuals_vector))
print(jnp.std(residuals_vector))

print("saving")
bytes_output = serialization.to_bytes(params)
with open("params.bin", "wb") as f:
    f.write(bytes_output)

dict_output = serialization.to_state_dict(params)
with open("params_dict.txt", "w") as f:
    f.write(str(dict_output))

print("saved")
