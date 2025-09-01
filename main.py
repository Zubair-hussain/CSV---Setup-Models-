import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import joblib
import os
import logging
import io
import base64
from huggingface_hub import InferenceClient
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit-app")

# Configuration
HF_TOKEN = "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"  # Your Hugging Face token
SUBSAMPLE_SIZE = 500  # Default to 500 samples for testing
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Lightweight Autoencoder (matches Colab script)
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

# Helper Functions
def load_models(imputer_path="simple_imputer.pkl", autoencoder_path="autoencoder.pth"):
    """Load imputer and autoencoder models."""
    try:
        imputer = joblib.load(imputer_path)
        logger.info(f"Loaded imputer from {imputer_path}")
    except Exception as e:
        logger.error(f"Error loading imputer: {e}")
        st.error(f"Failed to load imputer: {e}")
        return None, None
    try:
        input_dim = 30  # Based on creditcard.csv numeric columns (excluding 'class')
        model = SmallAutoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(autoencoder_path, map_location=torch.device('cpu')))
        logger.info(f"Loaded autoencoder from {autoencoder_path}")
        return imputer, model
    except Exception as e:
        logger.error(f"Error loading autoencoder: {e}")
        st.error(f"Failed to load autoencoder: {e}")
        return imputer, None

def load_and_subsample(file_path, subsample_size=SUBSAMPLE_SIZE):
    """Load and subsample dataset."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        if subsample_size and subsample_size < len(df):
            df = df.sample(n=subsample_size, random_state=RANDOM_SEED).reset_index(drop=True)
            logger.info(f"Subsampled to: {df.shape}")
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Failed to load dataset: {e}")
        return None

def impute_missing_values(df, imputer):
    """Impute missing values using loaded imputer."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns found; skipping imputation")
            return df
        df_num = df[numeric_cols].copy()
        df_num_imputed = pd.DataFrame(imputer.transform(df_num), columns=df_num.columns)
        df_imputed = df.copy()
        df_imputed[numeric_cols] = df_num_imputed
        logger.info("Imputation completed")
        return df_imputed
    except Exception as e:
        logger.error(f"Imputation error: {e}")
        st.error(f"Imputation failed: {e}")
        return df

def detect_anomalies_autoencoder(df, model, class_col='class'):
    """Detect anomalies using loaded autoencoder."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(class_col, errors='ignore')
        X = df[numeric_cols].values.astype(np.float32)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X)
            recon = model(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
        threshold = np.percentile(errors, 95)
        anomaly_indices = np.where(errors > threshold)[0].tolist()
        logger.info(f"Anomaly detection: threshold={threshold:.6g}, anomalies_found={len(anomaly_indices)}")
        return anomaly_indices, errors, threshold, numeric_cols.tolist()
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        st.error(f"Anomaly detection failed: {e}")
        return [], np.zeros(len(df)), 0, []

def get_deepseek_suggestions(sample_data, anomaly_info):
    """Call DeepSeek API for cleaning suggestions."""
    try:
        client = InferenceClient(model="deepseek/deepseek-coder-6.7b-instruct", token=HF_TOKEN)
        prompt = (
            f"Analyze this credit card fraud dataset sample: {sample_data[:3]}. "
            f"Anomaly detection results: {anomaly_info}. "
            "Suggest specific data cleaning or preprocessing steps to improve dataset quality and anomaly detection accuracy. "
            "Focus on handling outliers, scaling, or feature engineering. If unsure, say 'Insufficient information'."
        )
        response = client.generate(prompt, max_tokens=500, temperature=0.7)
        logger.info("DeepSeek suggestions received")
        return response
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        st.error(f"DeepSeek API failed: {e}")
        return "DeepSeek suggestions unavailable"

def generate_visualizations(df, errors, threshold, numeric_cols):
    """Generate base64 visualizations for Streamlit."""
    visuals = {}
    plt.figure(figsize=(10, 4))
    sns.heatmap(df[numeric_cols].isnull().iloc[:500], cbar=False)
    plt.title("Missing Values (First 500 Rows)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visuals['missing_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(errors)), errors, s=6)
    plt.axhline(threshold, ls='--', color='red')
    plt.title("Reconstruction Errors")
    plt.xlabel("Row Index")
    plt.ylabel("Error")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visuals['error_scatter'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(12, 5))
    sample_for_box = df[numeric_cols].iloc[:1000]
    sns.boxplot(data=sample_for_box, orient='h')
    plt.title("Numeric Distributions (First 1000 Rows)")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visuals['boxplot'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    logger.info("Visualizations generated")
    return visuals

# Streamlit App
def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    st.title("Credit Card Fraud Detection App")
    st.write("Upload a dataset or use the default cleaned dataset to detect anomalies and view data cleaning suggestions.")

    # File paths
    default_dataset = "cleaned_creditcard.csv"
    imputer_path = "simple_imputer.pkl"  # Update if using iterative_imputer.pkl
    autoencoder_path = "autoencoder.pth"

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file (optional)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded dataset shape: {df.shape}")
    else:
        if os.path.exists(default_dataset):
            df = load_and_subsample(default_dataset, SUBSAMPLE_SIZE)
            st.write(f"Using default dataset with {SUBSAMPLE_SIZE} samples, shape: {df.shape}")
        else:
            st.error("Default dataset (cleaned_creditcard.csv) not found. Please upload a CSV.")
            return

    if df is None:
        st.error("Failed to load dataset.")
        return

    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Load models
    imputer, model = load_models(imputer_path, autoencoder_path)
    if imputer is None or model is None:
        return

    # Impute missing values
    st.subheader("Imputation")
    df_imputed = impute_missing_values(df, imputer)
    st.write("Missing values imputed. Check heatmap for confirmation.")

    # Anomaly detection
    st.subheader("Anomaly Detection")
    anomaly_indices, errors, threshold, numeric_cols = detect_anomalies_autoencoder(df_imputed, model)
    anomaly_info = f"{len(anomaly_indices)} anomalies detected (indices sample: {anomaly_indices[:20]})"
    st.write(anomaly_info)

    # DeepSeek suggestions
    st.subheader("DeepSeek Cleaning Suggestions")
    sample_data = df_imputed.head(5).to_dict(orient='records')
    deepseek_suggestions = get_deepseek_suggestions(sample_data, anomaly_info)
    st.write(deepseek_suggestions)

    # Visualizations
    st.subheader("Visualizations")
    visuals = generate_visualizations(df_imputed, errors, threshold, numeric_cols)
    for key, b64 in visuals.items():
        st.write(f"**{key.replace('_', ' ').title()}**")
        st.image(base64.b64decode(b64))

    # Save and download processed dataset
    output_csv = io.StringIO()
    df_imputed.to_csv(output_csv, index=False)
    st.download_button(
        label="Download Processed Dataset",
        data=output_csv.getvalue(),
        file_name="processed_creditcard.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
