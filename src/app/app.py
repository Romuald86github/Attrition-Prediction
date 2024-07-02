import os
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the best model from the MLflow registry
model_name = "RandomForestClassifier"
model_version = "1"
model_path = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_path)

# Load the preprocessing pipeline
preprocessing_pipeline_path = "models:/preprocessing_pipeline/1"
preprocessing_pipeline = mlflow.sklearn.load_model(preprocessing_pipeline_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    X_preprocessed = preprocessing_pipeline.transform(df)
    prediction = model.predict(X_preprocessed)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)