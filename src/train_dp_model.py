import pandas as pd
import tensorflow as tf
import tensorflow_privacy
from sklearn.model_selection import train_test_split
import json
import os

import mlflow
import mlflow.tensorflow

from privacy_metrics import compute_epsilon


# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("data/healthcare.csv")

# Remove irrelevant columns
data = data.drop(columns=[
    "Name",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Date of Admission",
    "Discharge Date"
])


# -----------------------------
# Encode categorical features
# -----------------------------
data = pd.get_dummies(data)


# -----------------------------
# Define target
# -----------------------------
target = "Test Results_Abnormal"

X = data.drop(columns=[
    "Test Results_Abnormal",
    "Test Results_Inconclusive",
    "Test Results_Normal"
])

y = data[target]


# -----------------------------
# Save feature schema
# -----------------------------
os.makedirs("models", exist_ok=True)

feature_columns = list(X.columns)

with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)


# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Convert to tensors
# -----------------------------
X_train = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test.values, dtype=tf.float32)

y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)


# -----------------------------
# Differential Privacy optimizer
# -----------------------------
optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=1,
    learning_rate=0.001
)


# -----------------------------
# Build model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# MLflow experiment tracking
# -----------------------------
mlflow.set_experiment("privacy_ml_training")

with mlflow.start_run():

    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("noise_multiplier", 1.1)

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    val_accuracy = history.history["val_accuracy"][-1]
    mlflow.log_metric("validation_accuracy", val_accuracy)


    # -----------------------------
    # Compute privacy budget
    # -----------------------------
    dataset_size = len(X_train)

    epsilon = compute_epsilon(
        dataset_size=dataset_size,
        batch_size=32,
        noise_multiplier=1.1,
        epochs=10
    )

    mlflow.log_metric("epsilon", epsilon)

    print(f"Privacy Budget (ε): {epsilon}")

    # Log model to MLflow
    mlflow.tensorflow.log_model(model, "model")


# -----------------------------
# Save trained model locally
# -----------------------------
model.save("models/private_model.keras")

print("Model training complete and saved.")