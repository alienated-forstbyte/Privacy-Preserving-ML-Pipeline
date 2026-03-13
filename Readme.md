```markdown
# Privacy-Preserving ML Pipeline (Differential Privacy + MLOps)

A production-style **machine learning pipeline that trains privacy-preserving models and deploys them with monitoring and CI/CD automation**.

This project demonstrates how to combine **Differential Privacy**, **experiment tracking**, **reproducible pipelines**, **model serving**, and **observability** in a modern MLOps stack.

---
```
## Architecture

```
Dataset
│
▼
DVC Pipeline
│
▼
Differential Privacy Training (TensorFlow Privacy)
│
▼
MLflow Experiment Tracking
│
▼
Saved Model (.keras)
│
▼
FastAPI Model API
│
▼
Prometheus Metrics
│
▼
Grafana Dashboards
│
▼
Docker Deployment
│
▼
GitHub Actions CI/CD
```
```

---

## Features

### Differentially Private ML Training
- Uses **DP-SGD via TensorFlow Privacy**
- Protects training data from reconstruction attacks
- Tracks **privacy budget ε**

### Reproducible ML Pipelines
- Pipeline orchestration with **DVC**
- Deterministic model training

### Experiment Tracking
MLflow logs:

- Hyperparameters  
- Validation accuracy  
- Privacy budget ε  
- Model artifacts  

### Model Serving

- REST API built with **FastAPI**
- Supports real-time inference requests

### Monitoring

- **Prometheus** collects API metrics
- **Grafana** visualizes model performance

### Containerized Deployment

The entire stack runs with **Docker Compose**.

### CI/CD

GitHub Actions pipeline automatically:

- runs the training pipeline
- builds Docker image
- pushes image to registry

---

## Project Structure

```
```
Privacy-Preserving-ML-Pipeline
│
├── data/                    # Dataset
├── models/                  # Trained model artifacts
│
├── src/
│   └── train_dp_model.py    # Differential privacy training pipeline
│
├── deployment/
│   └── api.py               # FastAPI inference service
│
├── monitoring/
│   └── prometheus.yml       # Prometheus configuration
│
├── tests/                   # API tests
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
```
````

---

## Training the Model

Run the reproducible training pipeline:

bash
dvc repro
````

Outputs:    

```
models/private_model.keras
models/feature_columns.json
```

Example output:

```
Privacy Budget (ε): 0.58
Model training complete and saved.
```

---

## Running the API

Start the model server:

```bash
uvicorn deployment.api:app --reload
```

Open API docs:

```
http://localhost:8000/docs
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
"Age":45,
"Gender":"Male",
"Blood_Type":"O+",
"Medical_Condition":"Diabetes"
}'
```

---

## Monitoring

### Prometheus

Metrics endpoint:

```
http://localhost:8000/metrics
```

Example metric:

```
model_predictions_total
```

---

### Grafana

Open dashboard:

```
http://localhost:3000
```

Default credentials:

```
admin
admin
```

Add Prometheus datasource:

```
http://prometheus:9090
```

---

## Running the Full Stack

Run everything with Docker:

```bash
docker compose up --build
```

Services:

| Service     | URL                                            |
| ----------- | ---------------------------------------------- |
| FastAPI API | [http://localhost:8000](http://localhost:8000) |
| Prometheus  | [http://localhost:9090](http://localhost:9090) |
| Grafana     | [http://localhost:3000](http://localhost:3000) |

---

## CI/CD Pipeline

GitHub Actions automatically:

1. Runs the DVC pipeline
2. Trains the DP model
3. Builds Docker image
4. Pushes image to Docker Hub

Triggered on every push to **main**.

---

## Technologies Used

* Python
* TensorFlow
* TensorFlow Privacy
* DVC
* MLflow
* FastAPI
* Prometheus
* Grafana
* Docker
* GitHub Actions

---

## Example ML Metrics

Tracked metrics include:

* Validation accuracy
* Privacy budget ε
* Prediction request count
* API latency

---

## Why Differential Privacy?

Differential Privacy ensures the model **does not memorize individual training records**.

This is critical for sensitive domains:

* Healthcare
* Finance
* Government data

---

## Future Improvements

Potential extensions:

* automated model retraining
* data drift detection
* canary deployments
* Kubernetes deployment
* feature store integration

---

## License

MIT License
