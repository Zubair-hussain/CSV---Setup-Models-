from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import logging
from huggingface_hub import InferenceClient
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api-app")

# Config
HF_TOKEN = os.getenv("HF_TOKEN", "your_fallback_token")
SUBSAMPLE_SIZE = 200
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

app = FastAPI(title="Credit Card Fraud Detection API")

# Small Autoencoder
class SmallAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden1=32, hidden2=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load models
def load_models(imputer_path="simple_imputer.pkl", autoencoder_path="autoencoder.pth"):
    imputer = joblib.load(imputer_path)
    model = SmallAutoencoder(input_dim=30)
    model.load_state_dict(torch.load(autoencoder_path, map_location=torch.device("cpu")))
    return imputer, model

imputer, model = load_models()

# API Endpoints
@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = imputer.transform(df[numeric_cols])

        # Anomaly detection
        X = df[numeric_cols].values.astype(np.float32)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X)
            recon = model(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
        threshold = np.percentile(errors, 95)
        anomalies = np.where(errors > threshold)[0].tolist()

        # Call DeepSeek
        client = InferenceClient(model="deepseek/deepseek-coder-6.7b-instruct", token=HF_TOKEN)
        sample_data = df.head(5).to_dict(orient="records")
        anomaly_info = f"{len(anomalies)} anomalies detected (sample: {anomalies[:20]})"
        prompt = (
            f"Analyze this credit card dataset sample: {sample_data[:3]}. "
            f"Anomaly detection: {anomaly_info}. "
            "Suggest preprocessing steps to improve quality."
        )
        deepseek_response = client.generate(prompt, max_tokens=300).generated_text

        return {
            "rows": len(df),
            "anomalies": anomalies,
            "threshold": float(threshold),
            "deepseek_suggestions": deepseek_response
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
