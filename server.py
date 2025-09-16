import subprocess
import os

# Run Streamlit app
def start_streamlit():
    os.environ["STREAMLIT_SERVER_PORT"] = os.getenv("PORT", "8501")
    subprocess.run(["streamlit", "run", "main.py", "--server.headless", "true", "--server.address", "0.0.0.0"])

if __name__ == "__main__":
    start_streamlit()
