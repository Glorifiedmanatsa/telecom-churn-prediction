import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
# Load trained model
model = joblib.load("models/churn_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data and convert to float
    data = [float(x) for x in request.form.values()]
    
    # Correct shape
    features = np.zeros((1, model.n_features_in_))
    features[0, :len(data)] = data

    prob = model.predict_proba(features)[0][1]
    if prob > 0.7:
        risk = "High Risk ⚠️"
        color = "red"
    elif prob > 0.4:
        risk = "Medium Risk ⚡"
        color = "orange"
    else:
        risk = "Low Risk ✅"
        color = "green"

    return render_template(
        "result.html",   # ✅ FIXED (comma was missing)
        prediction=risk,
        probability=round(prob * 100, 2),
        color=color
    )

if __name__ == "__main__":
    if __name__ == "__main__":
     app.run(host="0.0.0.0", port=10000)