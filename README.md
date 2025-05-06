# CloudGuardian AI: Cloud-Native Threat Detection System

**Version:** 1.0
**Date:** May 3, 2025

## 1. Introduction

This project implements an AI/ML-based threat detection system for cloud environments, initially focusing on AWS CloudTrail logs. It analyzes log data to identify anomalous behaviors and potential security threats that might be missed by traditional methods. This serves as a portfolio piece demonstrating skills in cloud security, data science, and AI/ML application in cybersecurity.

## 2. Goals

*   Detect anomalous activities in AWS CloudTrail logs using Machine Learning (Isolation Forest).
*   Simulate realistic normal and attack-like CloudTrail activity.
*   Process raw logs into features suitable for ML.
*   Train and evaluate an anomaly detection model.
*   Implement advanced analysis techniques including graph-based approaches for modeling entity relationships.
*   Visualize detected anomalies via a basic dashboard.

## 3. Architecture

```
+---------------------+      +---------------------+      +---------------------+      +---------------------+      +---------------------+| Log Simulation      | ---> | Data Processing     | ---> | ML Engine           | ---> | Model Artifacts     | ---> | Visualization       || (generate_sample_  |      | (process_logs.py)   |      | (train_predict.py)  |      | (models/*.joblib,   |      | (dashboard.py)      ||    logs.py)         |      | - Parse JSON/GZ     |      | - Load Data         |      |  *.json)            |      | - Load Data         || - Normal Activity   |      | - Flatten Records   |      | - Prepare Features  |      +---------------------+      | - Load Model        || - Attack Simulation |      | - Feature Engineer  |      | - Split Train/Test  |             |              | - Predict Anomalies || - Add Labels        |      |   (Time, Error, UA, |      | - Train IsoForest   |             |              | - Display Alerts    ||                     |      |    Freq Analysis)   |      | - Evaluate Model    |             v              | - Show Scores       ||                     |      | - Save to Parquet   |      | - Save Artifacts    |      +---------------------+      |                     |
+---------------------+      +---------------------+      | - Predict Anomalies |                                    +---------------------+
                                                     +---------------------+
```

*   **Log Simulation:** Generates `.json.gz` CloudTrail-like logs with normal and simulated attack events, including labels.
*   **Data Processing:** Parses logs, extracts features (time, user, IP, frequency counts, etc.), and saves data as a Parquet file.
*   **ML Engine:** Loads processed data, prepares features (including one-hot encoding), splits into train/test sets, trains an Isolation Forest model, evaluates it using labels, and saves the model, encoder, and feature names.
*   **Model Artifacts:** Saved `joblib` files for the model and encoder, and a JSON file for feature names.
*   **Visualization:** A Streamlit dashboard loads data and uses the saved model artifacts to predict and display anomalies.

## 4. Technology Stack

*   **Language:** Python 3.x
*   **Data Handling:** Pandas, NumPy
*   **AI/ML:** Scikit-learn (IsolationForest, OneHotEncoder, metrics), TensorFlow/Keras, NetworkX/DGL/PyTorch Geometric (for graph analysis)
*   **Visualization:** Streamlit, Plotly
*   **Serialization:** Joblib, JSON
*   **Environment:** Venv

## 5. Project Structure

```
/
|-- data/
|   |-- processed/              # Processed data (Parquet)
|   |-- raw_logs/               # Raw simulated logs (.json.gz)
|-- models/                     # Saved ML model artifacts
|-- notebooks/                  # (Optional) Jupyter notebooks for exploration
|-- src/
|   |-- data_processing/        # Log processing script
|   |   `-- process_logs.py
|   |-- ml_engine/              # ML training and prediction script
|   |   `-- train_predict.py
|   |-- simulation/             # Log generation script
|   |   `-- generate_sample_logs.py
|   |-- visualization/          # Streamlit dashboard script
|   |   `-- dashboard.py
|-- tests/                      # Unit/Integration tests (Placeholder)
|-- README.md                   # This file
|-- requirements.txt            # Python dependencies
|-- project.md                  # Original project plan
|-- venv/                       # Virtual environment
```

## 6. Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **AWS Credentials Setup (if interacting with real AWS services):**
    
    This project uses boto3 for AWS interactions. Configure your AWS credentials using one of these methods:
    
    - **Option 1: AWS CLI configuration**
      ```bash
      aws configure
      # Follow prompts to enter your AWS Access Key ID, Secret Access Key, region, etc.
      ```
    
    - **Option 2: Environment variables**
      ```bash
      export AWS_ACCESS_KEY_ID="your_access_key"
      export AWS_SECRET_ACCESS_KEY="your_secret_key"
      export AWS_DEFAULT_REGION="your_region"
      ```
    
    - **Option 3: For local development, create a .env file** (requires uncommenting python-dotenv in requirements.txt)
      ```
      # .env file (DO NOT COMMIT THIS FILE)
      AWS_ACCESS_KEY_ID=your_access_key
      AWS_SECRET_ACCESS_KEY=your_secret_key
      AWS_DEFAULT_REGION=your_region
      ```
    
    **Note:** For the demo with simulated data, AWS credentials are not required.

## 7. Usage Workflow

Run these commands from the project root directory.

1.  **Generate Sample Logs:** (Creates labeled data in `data/raw_logs/`)
    ```bash
    python src/simulation/generate_sample_logs.py --output-dir data/raw_logs --num-files 5 --events-per-file 200 --attack-ratio 0.1 --compress
    ```
    *(Adjust parameters like `--num-files`, `--events-per-file`, `--attack-ratio` as needed)*

2.  **Process Logs:** (Parses raw logs, engineers features, saves to `data/processed/processed_cloudtrail_logs.parquet`)
    ```bash
    python src/data_processing/process_logs.py --raw-dir data/raw_logs --processed-dir data/processed
    ```

3.  **Train Model & Evaluate:** (Loads processed data, trains Isolation Forest, evaluates, saves artifacts to `models/`)
    ```bash
    python src/ml_engine/train_predict.py --data-path data/processed/processed_cloudtrail_logs.parquet --model-dir models --test-size 0.3
    ```

4.  **Run Visualization Dashboard:** (Starts the Streamlit app)
    ```bash
    streamlit run src/visualization/dashboard.py
    ```
    *   Use the sidebar in the dashboard to confirm the paths to the processed data (`data/processed/processed_cloudtrail_logs.parquet`) and model directory (`models/`).
    *   Interact with the dashboard to view detected anomalies and scores.
    *   **Note:** There might be a `RuntimeError: Event loop is closed` issue with Streamlit depending on the environment/versions. Further investigation may be needed if the dashboard doesn't run correctly.

## 8. Evaluation Results (Example)

Based on the last run with simulated data (70/30 train/test split):

*   **Precision:** ~0.19 (Many false positives)
*   **Recall:** ~0.99 (Most true attacks detected)
*   **F1-Score:** ~0.31
*   **Confusion Matrix:**
    ```
    [[  5 335]  # True Normal (TN=5, FP=335)
     [  1  77]]  # True Attack (FN=1, TP=77)
    ```

**Interpretation:** The default Isolation Forest model is good at catching anomalies (high recall) but flags many normal events as anomalous (low precision). Tuning the `contamination` parameter or exploring different models/features could improve precision.

## 9. Future Enhancements

*   Tune model parameters (e.g., `contamination`).
*   Implement more advanced feature engineering (API sequences, sessionization).
*   Add other ML models (Autoencoder, LSTM, XGBoost).
*   Implement Graph-Based Analysis:
    * Convert cloud infrastructure and activity into graph structures
    * Apply Graph Neural Networks (GNNs) to model entity relationships
    * Use graph algorithms for centrality measures and community detection
    * Implement temporal graph analysis to track evolving relationships
    * Integrate with security knowledge graphs for context-aware detection
*   Calculate AUC score.
*   Fix potential Streamlit runtime errors.
*   Add comprehensive tests.
*   Integrate more data sources (VPC Flow Logs).

## 10. Security Considerations

*   **Data Privacy:** All data in this repository is simulated/synthetic. No real AWS CloudTrail logs are included.
*   **Credentials:** No AWS credentials or API keys are stored in this codebase. When deploying:
    * Use environment variables or AWS credential providers
    * Never commit .env files or credentials to version control
    * Consider using IAM roles when deploying to AWS infrastructure
*   **Dependencies:** Regularly update dependencies to address security vulnerabilities.