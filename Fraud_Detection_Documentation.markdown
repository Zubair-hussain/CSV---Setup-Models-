# Fraud Detection Code Documentation

## Overview
This project implements a fraud detection system for credit card transactions using the `mlg-ulb/creditcardfraud` dataset. It includes data preprocessing, missing value imputation, anomaly detection with a PyTorch Autoencoder, AI-driven suggestions via Gemini API, and visualizations. The code supports two phases:
- **Training**: Performed in Google Colab with free T4 GPU access (2-3 runs, ~8-16 minutes each for 10,000 rows). Saves models (`imputer_model.pkl`, `autoencoder.pth`) and cleaned data (`cleaned_creditcard.csv`).
- **Inference**: Lightweight, CPU-based, suitable for deployment on Vercel/Railway or local execution on low-resource devices (e.g., 4GB RAM laptop). Processes new CSVs, imputes missing values, detects anomalies, and generates visualizations (~1-2 minutes for 2,000 rows).

The system is designed for integration with a Next.js portfolio website (e.g., `https://zubair-hussain-portfolio-hpne.vercel.app/`) via a FastAPI backend.

## Purpose
- **Objective**: Detect fraudulent transactions in credit card data by identifying anomalies using an Autoencoder, imputing missing values with IterativeImputer, and providing actionable suggestions via Gemini/SerpAPI.
- **Use Case**: Testing phase for a portfolio project, with deployment on Vercel or Railway for real-time fraud detection via a web interface.

## Prerequisites
- **Python**: 3.8+
- **Dependencies** (install via `pip install -r requirements.txt`):
  ```
  pandas
  numpy
  requests
  python-dotenv
  scikit-learn
  torch
  matplotlib
  seaborn
  datasets
  joblib
  fastapi
  uvicorn
  python-multipart
  kagglehub[hf-datasets]
  ```
- **Hardware**:
  - **Training**: Google Colab with T4 GPU (~12GB RAM, ~15GB VRAM).
  - **Inference**: CPU-only, ~500MB-1GB RAM (suitable for 4GB RAM laptop or serverless platforms like Vercel/Railway).
- **Credentials**:
  - **Kaggle API**: For dataset access (`KAGGLE_USERNAME`, `KAGGLE_KEY`). Store in Colab Secrets or `~/.kaggle/kaggle.json`.
  - **Gemini API**: For AI suggestions (`GEMINI_API_KEY`).
  - **SerpAPI**: For web search fallback (`SERPAPI_KEY`).
- **Files** (for inference/deployment):
  - `imputer_model.pkl`: Trained IterativeImputer model.
  - `autoencoder.pth`: Trained Autoencoder weights.
  - `cleaned_creditcard.csv`: Sample cleaned dataset (optional).

## Code Structure
The code is modular, with functions for data loading, preprocessing, anomaly detection, AI suggestions, visualizations, and a FastAPI endpoint for deployment.

### 1. Setup and Configuration
- **Purpose**: Install dependencies, configure API keys, and set global parameters.
- **Key Variables**:
  - `SUBSAMPLE_SIZE`: Number of rows to process (default: 10,000 for training, 2,000 for inference to fit 4GB RAM).
  - `ARTIFICIAL_MISSING_PCT`: Percentage of missing values to introduce (default: 10%).
  - `INSTRUCTIONS`: Prompt for Gemini API ("Detect fraud anomalies and suggest fixes for inconsistencies").
  - `GEMINI_API_KEY`, `SERPAPI_KEY`: API keys for suggestions (store in `.env` or Colab Secrets).
  - `KAGGLE_USERNAME`, `KAGGLE_KEY`: For KaggleHub dataset access.
- **Security**:
  - Use Colab Secrets or `.env` for API keys.
  - Revoke any exposed Kaggle keys (e.g., `17f6be22606c3839c2e942f3dbbd83f1`) via Kaggle > Settings > API > Expire Token.
- **Code**:
  ```python
  GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC_mpuFY_C4jKQq_RGLzPmR-X70rjmtM5c")
  SERPAPI_KEY = os.getenv("SERPAPI_KEY", "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e")
  try:
      KAGGLE_USERNAME = userdata.get('KAGGLE_USERNAME')
      KAGGLE_KEY = userdata.get('KAGGLE_KEY')
  except:
      KAGGLE_USERNAME = input("Enter Kaggle username: ")
      KAGGLE_KEY = input("Enter Kaggle API key: ")
  os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
  os.environ['KAGGLE_KEY'] = KAGGLE_KEY
  ```

### 2. Data Loading (`load_and_validate_csv`)
- **Purpose**: Load `creditcard.csv` from KaggleHub (`mlg-ulb/creditcardfraud`) or local file, subsample, and validate columns.
- **Input**: `subsample_size` (e.g., 10,000 for training, 2,000 for inference).
- **Output**: Pandas DataFrame with ~31 columns (e.g., `Time`, `V1`-`V28`, `Amount`, `Class`).
- **Validation**: Checks for expected columns; logs warning if mismatch.
- **Resources**: ~200MB RAM for 10,000 rows, ~50MB for 2,000 rows.
- **Code**:
  ```python
  def load_and_validate_csv(subsample_size=None):
      logger.info("Loading dataset from KaggleHub...")
      hf_dataset = kagglehub.load_dataset(KaggleDatasetAdapter.HUGGING_FACE, "mlg-ulb/creditcardfraud", "creditcard.csv")
      df = hf_dataset.to_pandas()
      if subsample_size:
          df = df.sample(n=min(subsample_size, len(df)), random_state=42)
      logger.info(f"Loaded CSV with shape: {df.shape}")
      return df
  ```
- **Inference Variant**: Loads local CSV (e.g., `cleaned_creditcard.csv`).

### 3. Preprocessing (`introduce_artificial_missing`, `impute_missing_values`)
- **Introduce Missing Values**:
  - Randomly sets 10% of numeric values to NaN for testing imputation.
  - Resources: ~200MB RAM (10,000 rows).
  - Code:
    ```python
    def introduce_artificial_missing(df, pct=0.1):
        mask = np.random.rand(*df.shape) < pct
        df_numeric = df.select_dtypes(include=np.number)
        df_numeric[mask[:, :len(df_numeric.columns)]] = np.nan
        df[df_numeric.columns] = df_numeric
        logger.info(f"Introduced {pct*100}% missing values.")
        return df
    ```
- **Impute Missing Values**:
  - Uses `IterativeImputer` with `RandomForestRegressor` (10 estimators) to impute numeric columns.
  - Training: Fits and saves `imputer_model.pkl` (~50MB).
  - Inference: Loads `imputer_model.pkl` for transformation.
  - Resources: ~1-2GB RAM (training, 10,000 rows), ~500MB (inference, 2,000 rows).
  - Runtime: ~2-5 minutes (training), ~20-30 seconds (inference).
  - Code:
    ```python
    def impute_missing_values(df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        if numeric_cols.size > 0:
            imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            joblib.dump(imputer, "/content/imputer_model.pkl")
        return df
    ```

### 4. Anomaly Detection (`Autoencoder`, `detect_anomalies`)
- **Autoencoder Model**:
  - PyTorch neural network with encoder (30→32→16) and decoder (16→32→30) for anomaly detection.
  - Trained on non-fraudulent data (`Class=0`) to reconstruct normal patterns.
  - Code:
    ```python
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
        def forward(self, x):
            return self.decoder(self.encoder(x))
    ```
- **Detection**:
  - Training: Uses GPU (if available) for 20 epochs, batch size 64. Saves `autoencoder.pth`.
  - Inference: Loads `autoencoder.pth`, computes reconstruction errors, flags anomalies (errors > 95th percentile).
  - Resources: ~1-2GB VRAM (training), ~500MB RAM (inference).
  - Runtime: ~5-10 minutes (training, Colab T4 GPU), ~10-20 seconds (inference, CPU).
  - Code:
    ```python
    def detect_anomalies(df):
        numeric_cols = df.select_dtypes(include=np.number).columns.drop('Class', errors='ignore')
        X = df[numeric_cols].values
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Autoencoder(input_dim=len(numeric_cols)).to(device)
        # Training or inference logic
        torch.save(model.state_dict(), "/content/autoencoder.pth")
        return anomalies, errors, threshold, anomaly_info
    ```

### 5. AI Suggestions (`get_ai_suggestions`, `web_search_fallback`)
- **Gemini API**:
  - Sends CSV sample (5 rows), anomaly info, and instructions to Gemini for fraud detection suggestions.
  - Runtime: ~5-10 seconds (network-dependent).
  - Code:
    ```python
    def get_ai_suggestions(sample_data, anomaly_info, instructions):
        ai_prompt = f"Analyze this CSV sample: {sample_data}. {anomaly_info}. ..."
        response = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", ...)
        return suggestions
    ```
- **SerpAPI Fallback**:
  - If Gemini returns "unknown," queries Google for fraud detection techniques.
  - Runtime: ~5-10 seconds.
  - Code:
    ```python
    def web_search_fallback(suggestions, df_columns):
        if 'unknown' in suggestions.lower():
            query = f"credit card fraud detection techniques for anomalies in columns like {', '.join(df_columns)}"
            url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={SERPAPI_KEY}"
            return suggestions + "\nWeb suggestions: " + web_results
        return suggestions
    ```

### 6. Visualizations (`generate_visualizations`)
- **Purpose**: Generate base64-encoded PNGs (heatmap, scatter plot, boxplot) for missing values, anomaly errors, and data distribution.
- **Resources**: ~500MB-1GB RAM (10,000 rows), ~200MB (2,000 rows).
- **Runtime**: ~20-40 seconds (10,000 rows), ~10-20 seconds (2,000 rows).
- **Code**:
  ```python
  def generate_visualizations(df, errors, threshold, anomalies):
      visuals = {}
      plt.figure(figsize=(8, 4))
      sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
      plt.title("Missing Values Heatmap")
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      visuals['missing_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
      plt.close()
      # Similar for scatter, boxplot
      return visuals
  ```

### 7. FastAPI Endpoint (`/predict`)
- **Purpose**: Accepts CSV uploads, runs inference (imputation, anomaly detection, suggestions, visualizations), and returns results.
- **Input**: CSV file (multipart form data).
- **Output**: JSON with anomalies, suggestions, visuals, and cleaned CSV.
- **Code**:
  ```python
  @app.post("/predict")
  async def predict(file: UploadFile = File(...)):
      df = load_and_validate_csv(io.StringIO(contents.decode('utf-8')), SUBSAMPLE_SIZE)
      df = impute_missing_values(df)
      anomalies, errors, threshold, anomaly_info = detect_anomalies(df)
      suggestions = get_ai_suggestions(...)
      visuals = generate_visualizations(...)
      return {"anomalies": anomaly_info, "suggestions": suggestions, "visuals": visuals, "cleaned_csv": df.to_csv(index=False)}
  ```

## Usage
### Training (Colab, Free T4 GPU)
1. **Setup**:
   - Enable T4 GPU: Runtime > Change runtime type > T4 GPU.
   - Configure Kaggle API:
     - Colab Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`.
     - Or upload `kaggle.json`:
       ```python
       from google.colab import files
       files.upload()
       !mkdir -p ~/.kaggle
       !cp kaggle.json ~/.kaggle/
       !chmod 600 ~/.kaggle/kaggle.json
       ```
   - Set `SUBSAMPLE_SIZE = 10000`.
2. **Run**:
   - Execute the full code (from August 31, 2025, 10:27 PM PKT).
   - Runtime: ~8-16 minutes per run (2-3 runs total).
   - Outputs: `cleaned_creditcard.csv`, `imputer_model.pkl`, `autoencoder.pth`.
   - Download:
     ```python
     files.download("/content/imputer_model.pkl")
     files.download("/content/autoencoder.pth")
     files.download("/content/cleaned_creditcard.csv")
     ```
3. **Verify**:
   - Logs: `Using device: cuda`, `Loaded CSV with shape: (10000, 31)`, `Epoch X, Loss: Y`.
   - Visuals: Heatmap (no missing values post-imputation), scatter (anomaly errors), boxplot.

### Inference (Laptop, Vercel, or Railway)
1. **Setup**:
   - Copy `imputer_model.pkl`, `autoencoder.pth`, `cleaned_creditcard.csv` to project directory.
   - Set `SUBSAMPLE_SIZE = 2000` for 4GB RAM laptop or serverless platforms.
   - Install dependencies: `pip install -r requirements.txt`.
2. **Local Testing (Laptop)**:
   - Run: `uvicorn main:app --host 0.0.0.0 --port 8000`.
   - Test: `curl -X POST -F "file=@cleaned_creditcard.csv" http://localhost:8000/predict`.
   - Runtime: ~1-2 minutes (2,000 rows).
3. **Deployment (Vercel)**:
   - Push `main.py`, `requirements.txt`, models to GitHub.
   - In Vercel: Import repo, set env vars (`GEMINI_API_KEY`, `SERPAPI_KEY`), configure build (`pip install -r requirements.txt`, `uvicorn main:app --host 0.0.0.0 --port $PORT`).
   - Test: `curl -X POST -F "file=@cleaned_creditcard.csv" https://your-api.vercel.app/predict`.
4. **Deployment (Railway)**:
   - Push to GitHub, connect to Railway, set env vars.
   - Test: Similar to Vercel.

### Portfolio Integration
- **Frontend**: Add a Next.js page (`pages/fraud-detection.js`) to your portfolio (`https://zubair-hussain-portfolio-hpne.vercel.app/`).
- **Features**: Upload CSV, display anomalies, suggestions, and base64 visuals.
- **Example**:
  ```javascript
  const res = await fetch('https://your-api.vercel.app/predict', { method: 'POST', body: formData });
  const data = await res.json();
  // Display data.anomalies, data.suggestions, data.visuals
  ```
- **Styling**: Use Tailwind CSS or custom CSS (like `HeroSection`, August 29, 2025).

## Performance
- **Training (Colab T4 GPU)**:
  - Runtime: ~8-16 minutes (10,000 rows).
  - Resources: ~1-2GB VRAM, ~2-3GB RAM.
- **Inference (CPU, Vercel/Railway/Laptop)**:
  - Runtime: ~1-2 minutes (2,000 rows).
  - Resources: ~500MB-1GB RAM, no GPU.
- **Laptop (4GB RAM)**:
  - Training: Risky (crashes likely). Use Colab.
  - Inference: Feasible with `SUBSAMPLE_SIZE = 2000`, close background apps.

## Troubleshooting
- **KaggleHub 401 Unauthorized**: Verify `KAGGLE_USERNAME`, `KAGGLE_KEY` in Colab Secrets or `kaggle.json`.
- **API Errors**: Check Gemini/SerpAPI rate limits (Google Cloud Console/SerpAPI dashboard).
- **Memory Issues**: Reduce `SUBSAMPLE_SIZE` to 1,000 or add swap space (8GB recommended).
- **Visuals Not Displaying**: Ensure `%matplotlib inline` in Colab/Jupyter.
- **Deployment Errors**: Check Vercel/Railway logs for missing dependencies or env vars.

## Security Notes
- **Kaggle API**: Revoke exposed keys (e.g., `17f6be22606c3839c2e942f3dbbd83f1`) via Kaggle.
- **API Keys**: Store in `.env` or Vercel/Railway env vars, not in code.
- **Firebase Auth**: Add your Vercel domain to Firebase Console > Authentication > Authorized Domains to fix `auth/unauthorized-domain` (from August 31, 2025).

## Future Enhancements
- **Rate Limiting**: Add `slowapi` to FastAPI.
- **Authentication**: Integrate Firebase Auth (like `ClientComments`, August 28, 2025).
- **Monitoring**: Use Vercel/Railway logs for usage tracking.
- **Frontend**: Add animations (Framer Motion, like splash screen, August 29, 2025).

## Contact
For questions, contact the developer via portfolio comments (`https://zubair-hussain-portfolio-hpne.vercel.app/`) or GitHub issues.