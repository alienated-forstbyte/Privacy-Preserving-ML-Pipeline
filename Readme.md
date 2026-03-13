Privacy-Preserving ML Pipeline (Differential Privacy + MLOps)
CI/CD Pipeline

GitHub Actions automatically:

Runs the DVC pipeline

Trains the DP model

Builds Docker image

Pushes image to Docker Hub

Triggered on every push to main.

Technologies Used

Python

TensorFlow

TensorFlow Privacy

DVC

MLflow

FastAPI

Prometheus

Grafana

Docker

GitHub Actions

Example ML Metrics

Tracked metrics include:

Validation accuracy

Privacy budget ε

Prediction request count

API latency

Why Differential Privacy?

Differential Privacy ensures that the model does not memorize individual training records.

This is critical in domains such as:

Healthcare

Finance

Government datasets

Future Improvements

Possible extensions:

automated model retraining

drift detection

canary deployments

Kubernetes deployment

feature store integration

License

MIT License.
