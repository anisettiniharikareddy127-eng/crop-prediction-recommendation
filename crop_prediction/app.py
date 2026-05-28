from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        result = model.predict(data)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)