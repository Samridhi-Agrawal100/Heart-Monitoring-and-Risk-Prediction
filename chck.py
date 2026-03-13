import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_ann_model-2.h5")

# load dataset
data = pd.read_csv(DATASET_PATH)

# features and target
X = data.drop("target", axis=1)
y = data["target"]

# load scaler
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# load model
model = load_model(MODEL_PATH)

# predictions
y_pred_prob = model.predict(X_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc = roc_auc_score(y, y_pred_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc)

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
