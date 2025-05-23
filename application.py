from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the model and scaler
ridge_model = pickle.load(open("ridge.pkl", "rb"))
standard_scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            temp = float(request.form.get("Temperature"))
            rh = float(request.form.get("RH"))
            winds = float(request.form.get("Winds"))
            rain = float(request.form.get("Rain"))
            ffmc = float(request.form.get("FFMC"))
            dmc = float(request.form.get("DMC"))
            isi = float(request.form.get("ISI"))
            classes = int(request.form.get("Classes"))
            region = int(request.form.get("Region"))

            input_data = [[temp, rh, winds, rain, ffmc, dmc, isi, classes, region]]
            scaled_data = standard_scaler.transform(input_data)
            result = ridge_model.predict(scaled_data)

            return render_template("home.html", result=round(result[0], 2))  # Rounded for cleaner display
        except Exception as e:
            return render_template("home.html", result="Error in input: " + str(e))
    else:
        return render_template("home.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
