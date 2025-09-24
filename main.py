from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import logging
from huggingface_hub import InferenceClient
from sklearn.impute import SimpleImputer
import io

# ---------------------------------------------------
# Logging setup
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fraud-api")

# ---------------------------------------------------
# API Keys (only from environment variables)
# ---------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
Enter_Kaggle_username = os.getenv("Enter_Kaggle_username")
Enter_Kaggle_API_key = os.getenv("Enter_Kaggle_API_key")
Enter_Hugging_Face_API_token = os.getenv("Enter_Hugging_Face_API_token")

# ---------------------------------------------------
# Config
# ---------------------------------------------------
SUBSAMPLE_SIZE = 200
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------
# Model Definition
# ---------------------------------------------------
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

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def load_models(imputer_path="simple_imputer.pkl", autoencoder_path="autoencoder.pth"):
    try:
        imputer = joblib.load(imputer_path)
        logger.info("Imputer loaded.")
    except Exception as e:
        logger.error(f"Imputer load error: {e}")
        imputer = None

    try:
        input_dim = 30
        model = SmallAutoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(autoencoder_path, map_location=torch.device("cpu")))
        logger.info("Autoencoder loaded.")
    except Exception as e:
        logger.error(f"Autoencoder load error: {e}")
        model = None

    return imputer, model

def impute_missing_values(df, imputer):
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_num_imputed = pd.DataFrame(imputer.transform(df[numeric_cols]), columns=numeric_cols)
        df[numeric_cols] = df_num_imputed
        return df
    except Exception as e:
        logger.error(f"Imputation error: {e}")
        return df

def detect_anomalies(df, model, class_col="class"):
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(class_col, errors="ignore")
        X = df[numeric_cols].values.astype(np.float32)
        model.eval()
        with torch.no_grad():
            recon = model(torch.tensor(X))
            errors = torch.mean((recon - torch.tensor(X)) ** 2, dim=1).numpy()
        threshold = np.percentile(errors, 95)
        anomalies = np.where(errors > threshold)[0].tolist()
        return {"threshold": float(threshold), "anomalies": anomalies, "errors": errors.tolist()}
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return {"error": str(e)}

def get_deepseek_suggestions(sample_data, anomaly_info):
    try:
        client = InferenceClient(model="deepseek/deepseek-coder-6.7b-instruct", token=Enter_Hugging_Face_API_token)
        prompt = (
            f"Dataset sample: {sample_data}. "
            f"Anomaly info: {anomaly_info}. "
            "Suggest preprocessing and cleaning steps."
        )
        resp = client.generate(prompt, max_tokens=300, temperature=0.7)
        return resp.generated_text
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return "DeepSeek suggestions unavailable."

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(title="Credit Card Fraud Detection API")

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.post("/process-file")
async def process_file(file: UploadFile = File(...)):
    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        logger.info(f"Uploaded dataset shape: {df.shape}")

        # Load models
        imputer, model = load_models()
        if imputer is None or model is None:
            return JSONResponse(content={"error": "Models not loaded."}, status_code=500)

        # Impute
        df = impute_missing_values(df, imputer)

        # Anomaly detection
        results = detect_anomalies(df, model)

        # DeepSeek suggestions
        suggestions = get_deepseek_suggestions(df.head(3).to_dict(orient="records"), results)

        return {
            "rows": len(df),
            "anomaly_results": results,
            "deepseek_suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
