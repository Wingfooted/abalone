from flax.serialization import from_bytes

from model import Model


def load_model(path: str):
    with open(path, "rb") as raw:
        byte_data = raw.read()
        model_structure = (10, 10, 10, 10)
        model = Model(model_structure)
        params = from_bytes(model, byte_data)
    return model, params


if __name__ == '__main__':
    import jax.random as random
    model, params = load_model("params.bin")
    x = random.normal(random.key(0), (10,))

    print(params)
    y = model.apply(params, x)
    print(y)
