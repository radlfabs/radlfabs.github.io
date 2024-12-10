import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from model_class import SimpleModel, MAPIEModel
from lookup import names_lookup


def process_json(json_):
    if isinstance(json_, list):
        json_ = json_[0]
    json_["location"] = names_lookup.get(json_["location"], json_["location"])
    df = pd.DataFrame(json_, index=[0])
    return df


def process_result(y_pred):
    return y_pred.round(2).astype(int).item()


def process_result_tuple(y_pis):
    return y_pis.round(2).astype(int).flatten().tolist()


model_path = Path("trained_models")
simple_path = model_path / "trained_ml_pipeline.pkl"
mapie_path = model_path / "trained_quantile_pipeline.pkl"
simple_model = joblib.load(simple_path)
mapie_model = joblib.load(mapie_path)

app = Flask(__name__)

@app.route("/predict",methods=["POST"])
def predict_simple():
    df = process_json(request.json)
    y_pred = simple_model.predict(df)
    y_pred = process_result(y_pred)
    return jsonify({"Prediction": y_pred})

@app.route("/predict-mapie",methods=["POST"])
def predict_mapie():
    df = process_json(request.json)
    y_pred, y_pis = mapie_model.predict_mapie(df)
    y_pred = process_result(y_pred)
    y_pis = process_result_tuple(y_pis)
    return jsonify({
        "Prediction": y_pred, 
        "PI": y_pis})

if __name__ == '__main__':
    app.run(debug=True)
