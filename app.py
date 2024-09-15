import random
import jax.numpy as jnp
from load import load_model
from flask import Flask, url_for, render_template, request, redirect

app = Flask(__name__)
abalone_model, abalone_params = load_model("params.bin")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/abalone/<val>", methods=["GET", "POST"])
def abalone(val):
    if request.method == "GET":
        return render_template("abalone.html", pred=val)

    if request.method == "POST":
        sex = request.form.get("sex")
        sex = sex if sex else random.choice(["M", "F", "I"])

        length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight = 0, 0, 0, 0, 0, 0, 0
        length = request.form.get("length")
        diameter = request.form.get("diameter")
        height = request.form.get("height")
        whole_weight = request.form.get("whole_weight")
        shucked_weight = request.form.get("shucked_weight")
        viscera_weight = request.form.get("viscera_weight")
        shell_weight = request.form.get("shell_weight")
        
        length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight = float(length), float(diameter), float(height), float(whole_weight), float(shucked_weight), float(viscera_weight), float(shell_weight)

        print(f"Sex: {sex}, Length: {length}, Diameter: {diameter}, Height: {height}, Whole Weight: {whole_weight}, ",
          f"Shucked Weight: {shucked_weight}, Viscera Weight: {viscera_weight}, Shell Weight: {shell_weight}")
        print(type(length))

        if sex == "M":
            s1, s2, s3 = 1, 0, 0
        if sex == "F":
            s1, s2, s3 = 0, 1, 0
        if sex == "I":
            s1, s2, s3 = 0, 0, 1

        x = jnp.array((length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, s1, s2, s3))
        abalone_pred = abalone_model.apply(abalone_params, x)

        return redirect(url_for("abalone", val=abalone_pred))


if __name__ == '__main__':
    app.run(debug=True,
            port=9000,
            host="0.0.0.0")
