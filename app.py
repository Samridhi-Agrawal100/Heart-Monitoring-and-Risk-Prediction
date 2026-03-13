from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask import jsonify
app = Flask(__name__)
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_ann_model-2.h5")

model = load_model("model/heart_ann_model-2.h5")

# load scaler
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    sex = float(request.form["sex"])
    bp = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = float(request.form["fbs"])
    hr = float(request.form["thalach"])

    features = np.array([[age, sex, bp, chol, fbs, hr]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    risk = float(round(prediction[0][0] * 100, 2))

    return jsonify({"risk":risk})


if __name__ == "__main__":
    app.run(debug=True)