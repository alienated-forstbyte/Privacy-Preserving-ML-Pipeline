from fastapi import FastAPI, Response
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import json
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

model = tf.keras.models.load_model("models/private_model.keras", compile=False)


class PredictionRequest(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str

with open("models/feature_columns.json") as f:
    feature_columns = json.load(f)

prediction_counter = Counter(
    "model_predictions_total",
    "Total number of predictions made"
)

request_latency = Histogram(
    "prediction_latency_seconds",
    "Time taken to run model prediction"
)


@app.post("/predict")
def predict(data: PredictionRequest):

    start = time.time()
    prediction_counter.inc()

    df = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender,
        "Blood Type": data.Blood_Type,
        "Medical Condition": data.Medical_Condition
    }])

    df = pd.get_dummies(df)

    # align with training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # convert to numeric tensor input
    features = df.astype("float32").values

    prediction = model.predict(features)

    latency = time.time() - start
    request_latency.observe(latency)

    return {"prediction": float(prediction[0][0])}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"}