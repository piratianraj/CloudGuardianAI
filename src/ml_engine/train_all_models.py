#!/usr/bin/env python3
# Script to train and evaluate all available models on the same dataset

import pandas as pd
import numpy as np
import os
import logging
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import model implementations
from train_predict import load_data, prepare_features, train_model as train_isolation_forest
from train_predict import save_model as save_isolation_forest, evaluate_model as evaluate_isolation_forest
from advanced_models import (
    train_autoencoder, save_autoencoder, predict_autoencoder,
    train_ocsvm, save_ocsvm, predict_ocsvm,
    train_supervised_model, save_supervised_model, predict_supervised,
    evaluate_model_performance, compare_models
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MODEL_TRAINER] - %(message)s')

# Define default paths
import os

# Get the absolute path based on the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

# Use absolute paths
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DEFAULT_DATA_TYPE = 'cloudtrail'  # Can be 'cloudtrail' or 'vpc'

# Define data paths based on data type
def get_data_path_for_type(data_type):
    """Returns the appropriate data path based on data type"""
    if data_type == 'vpc':
        return os.path.join(PROJECT_ROOT, 'data/processed/processed_vpc_flow_logs.parquet')
    else:  # default to cloudtrail
        return os.path.join(PROJECT_ROOT, 'data/processed/processed_cloudtrail_logs.parquet')
        
DEFAULT_PROCESSED_DATA_PATH = get_data_path_for_type(DEFAULT_DATA_TYPE)

def train_and_evaluate_all_models(data_path, model_dir, data_type, test_size=0.3):
    """
    Trains and evaluates all available models on the same dataset.
    
    Args:
        data_path: Path to the processed data file
        model_dir: Directory to save model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
        test_size: Proportion of data to use for testing
    """
    # 1. Load Data
    df = load_data(data_path, data_type)
    if df is None:
        logging.error("Data loading failed. Exiting.")
        return
    
    # Define target variable
    y = df['is_simulated_attack']
    
    # 2. Prepare Features
    X_prepared, feature_names, encoder = prepare_features(df.drop(columns=['is_simulated_attack']), data_type)
    if X_prepared is None or feature_names is None or encoder is None:
        logging.error("Feature preparation failed. Exiting.")
        return
    
    # 3. Split Data
    logging.info(f"Splitting data into train/test sets (Test size: {test_size})")
    try:
        # Check if there are enough samples for stratification
        if y.nunique() < 2:
            logging.warning("Only one class present in labels. Cannot stratify. Performing regular split.")
            stratify_param = None
        else:
            stratify_param = y
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        logging.info(f"Attack ratio in training set: {y_train.mean():.4f}")
        logging.info(f"Attack ratio in test set: {y_test.mean():.4f}")
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return
    
    # Dictionary to store evaluation results for each model
    model_results = {}
    
    # 4. Train and Evaluate Models
    
    # --- Isolation Forest ---
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING ISOLATION FOREST MODEL")
    logging.info("=" * 50)
    
    # Estimate contamination based on training data labels
    estimated_contamination = y_train.mean()
    # Ensure contamination is within the valid range (0, 0.5]
    if estimated_contamination <= 0 or estimated_contamination > 0.5:
        logging.warning(f"Estimated contamination {estimated_contamination:.4f} is outside (0, 0.5]. Using 'auto'.")
        contamination_level = 'auto'
    else:
        contamination_level = estimated_contamination
        logging.info(f"Using estimated contamination: {contamination_level:.4f}")
    
    # Train Isolation Forest
    iforest_model = train_isolation_forest(X_train, contamination_level=contamination_level)
    save_isolation_forest(iforest_model, encoder, feature_names, model_dir, data_type)
    
    # Predict and Evaluate
    iforest_predictions = iforest_model.predict(X_test)  # -1 for anomalies, 1 for normal
    iforest_scores = iforest_model.decision_function(X_test)
    
    # Convert Isolation Forest predictions (-1 anomaly, 1 normal) to (1 attack, 0 normal)
    iforest_predictions_converted = np.where(iforest_predictions == -1, 1, 0)
    
    # Evaluate
    evaluate_isolation_forest(y_test, iforest_predictions, iforest_scores, data_type)
    
    # Store results for comparison
    model_results['Isolation Forest'] = evaluate_model_performance(
        y_test, iforest_predictions_converted, -iforest_scores, "Isolation Forest"
    )
    
    # --- Autoencoder ---
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING AUTOENCODER MODEL")
    logging.info("=" * 50)
    
    # Train Autoencoder
    autoencoder_model, threshold, scaler = train_autoencoder(X_train, X_test, epochs=30)
    if autoencoder_model is not None:
        save_autoencoder(autoencoder_model, threshold, scaler, model_dir, data_type)
        
        # Predict and Evaluate
        autoencoder_predictions, autoencoder_scores = predict_autoencoder(autoencoder_model, threshold, scaler, X_test)
        
        # Evaluate
        model_results['Autoencoder'] = evaluate_model_performance(
            y_test, autoencoder_predictions, autoencoder_scores, "Autoencoder"
        )
    else:
        logging.error("Autoencoder training failed. Skipping evaluation.")
    
    # --- One-Class SVM ---
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING ONE-CLASS SVM MODEL")
    logging.info("=" * 50)
    
    # Use contamination estimate for nu parameter if available
    nu_param = min(max(0.01, estimated_contamination), 0.5) if estimated_contamination > 0 else 0.1
    
    # Train One-Class SVM
    ocsvm_model, ocsvm_scaler = train_ocsvm(X_train, nu=nu_param)
    if ocsvm_model is not None:
        save_ocsvm(ocsvm_model, ocsvm_scaler, model_dir, data_type)
        
        # Predict and Evaluate
        ocsvm_predictions, ocsvm_scores = predict_ocsvm(ocsvm_model, ocsvm_scaler, X_test)
        
        # Evaluate
        model_results['One-Class SVM'] = evaluate_model_performance(
            y_test, ocsvm_predictions, -ocsvm_scores, "One-Class SVM"
        )
    else:
        logging.error("One-Class SVM training failed. Skipping evaluation.")
    
    # --- Random Forest (Supervised) ---
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING RANDOM FOREST MODEL (SUPERVISED)")
    logging.info("=" * 50)
    
    # Train Random Forest
    rf_model, rf_scaler = train_supervised_model(X_train, y_train, model_type='random_forest')
    if rf_model is not None:
        save_supervised_model(rf_model, rf_scaler, model_dir, data_type, 'random_forest')
        
        # Predict and Evaluate
        rf_predictions, rf_probabilities = predict_supervised(rf_model, rf_scaler, X_test)
        
        # Evaluate
        model_results['Random Forest'] = evaluate_model_performance(
            y_test, rf_predictions, rf_probabilities, "Random Forest"
        )
    else:
        logging.error("Random Forest training failed. Skipping evaluation.")
    
    # --- XGBoost (Supervised) ---
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING XGBOOST MODEL (SUPERVISED)")
    logging.info("=" * 50)
    
    # Train XGBoost (using GradientBoostingClassifier as proxy)
    xgb_model, xgb_scaler = train_supervised_model(X_train, y_train, model_type='xgboost')
    if xgb_model is not None:
        save_supervised_model(xgb_model, xgb_scaler, model_dir, data_type, 'xgboost')
        
        # Predict and Evaluate
        xgb_predictions, xgb_probabilities = predict_supervised(xgb_model, xgb_scaler, X_test)
        
        # Evaluate
        model_results['XGBoost'] = evaluate_model_performance(
            y_test, xgb_predictions, xgb_probabilities, "XGBoost"
        )
    else:
        logging.error("XGBoost training failed. Skipping evaluation.")
    
    # 5. Compare Models
    if len(model_results) > 1:
        logging.info("\n" + "=" * 50)
        logging.info("MODEL COMPARISON")
        logging.info("=" * 50)
        compare_models(model_results, f"{data_type.capitalize()} Log Anomaly Detection Model Comparison")
    else:
        logging.warning("Not enough models were successfully trained for comparison.")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate all available models on the same dataset.")
    parser.add_argument('--data-type', default=DEFAULT_DATA_TYPE, choices=['cloudtrail', 'vpc'],
                        help=f"Type of log data to process ('cloudtrail' or 'vpc', default: {DEFAULT_DATA_TYPE})")
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save the trained model artifacts (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument('--test-size', type=float, default=0.3,
                        help="Proportion of data to use for the test set.")
    parser.add_argument('--data-path', default=None,
                        help="Path to the processed Parquet data file (if not specified, will be determined based on data-type)")
    
    args = parser.parse_args()
    
    # Determine data path based on data type if not explicitly provided
    data_path = args.data_path if args.data_path else get_data_path_for_type(args.data_type)
    logging.info(f"Using data path: {data_path} for data type: {args.data_type}")
    
    # Train and evaluate all models
    train_and_evaluate_all_models(
        data_path=data_path,
        model_dir=args.model_dir,
        data_type=args.data_type,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main()