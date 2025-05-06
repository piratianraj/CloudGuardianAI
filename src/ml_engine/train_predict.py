# Trains an Isolation Forest model for anomaly detection on processed CloudTrail or VPC Flow logs

import pandas as pd
import joblib
import os
import logging
import argparse
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [ML_ENGINE] - %(message)s')

# Define default paths relative to the script location
# DEFAULT_PROCESSED_DATA_PATH = '../../data/processed/processed_cloudtrail_logs.parquet'
DEFAULT_PROCESSED_DATA_PATH = '../../data/processed/processed_vpc_flow_logs.parquet' # Changed default to VPC logs
DEFAULT_MODEL_DIR = '../../models'
# Add a flag to specify data type
DEFAULT_DATA_TYPE = 'vpc' # Can be 'cloudtrail' or 'vpc'

def load_data(file_path, data_type):
    """Loads processed data from a Parquet file based on data_type."""
    logging.info(f"Loading processed {data_type} data from: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        # Ensure the label column exists (using the placeholder name from process_logs.py)
        label_col = 'is_simulated_attack'
        if label_col not in df.columns:
            logging.warning(f"Warning: '{label_col}' label column not found in data. Assuming all normal.")
            # Add a default label column if missing
            df[label_col] = 0
        else:
            # Convert boolean/object label to integer (1 for attack, 0 for normal)
            # Fill potential NaN/None values before converting to int
            df[label_col] = df[label_col].fillna(0).astype(int)
        return df
    except FileNotFoundError:
        logging.error(f"Error: Processed data file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def prepare_features(df, data_type):
    """Selects, encodes categorical features, and prepares the feature matrix based on data_type."""
    logging.info(f"Preparing features for {data_type} model training...")

    if data_type == 'cloudtrail':
        # --- CloudTrail Features ---
        numerical_features = [
            'event_hour', 'event_day_of_week',
            'events_per_user', 'events_per_ip', 'console_logins_per_user'
        ]
        categorical_features = ['eventName', 'user_type', 'awsRegion', 'is_error', 'user_agent_present']
    elif data_type == 'vpc':
        # --- VPC Flow Log Features ---
        numerical_features = [
            'srcport', 'dstport', 'protocol', 'packets', 'bytes', 'start', 'end', 'flow_duration'
            # Add more VPC features if engineered, e.g., bytes_per_packet
        ]
        # Ensure categorical features match those created in process_vpc_features
        categorical_features = ['action', 'log-status', 'protocol_name']
    else:
        logging.error(f"Unknown data_type '{data_type}' for feature preparation.")
        return None, None, None

    # Check for missing columns and handle gracefully
    available_numerical = [f for f in numerical_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]

    missing_numerical = set(numerical_features) - set(available_numerical)
    missing_categorical = set(categorical_features) - set(available_categorical)
    if missing_numerical:
        logging.warning(f"Missing numerical features: {missing_numerical}")
    if missing_categorical:
        logging.warning(f"Missing categorical features: {missing_categorical}")

    if not available_numerical and not available_categorical:
        logging.error("No features available for training.")
        return None, None, None

    # Handle potential NaN in numerical features (fill with 0 for simplicity here)
    df_numerical = df[available_numerical].fillna(0)

    # Handle categorical features using OneHotEncoder
    df_categorical = df[available_categorical].astype(str).fillna('missing') # Convert to string and fill NaNs
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cats = encoder.fit_transform(df_categorical)
    encoded_cat_feature_names = encoder.get_feature_names_out(available_categorical)

    # Combine numerical and encoded categorical features
    features_df = pd.concat([
        df_numerical.reset_index(drop=True),
        pd.DataFrame(encoded_cats, columns=encoded_cat_feature_names).reset_index(drop=True)
    ], axis=1)

    # Ensure all features are numeric and handle any remaining NaNs
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    feature_names = features_df.columns.tolist()
    logging.info(f"Prepared feature matrix with {len(feature_names)} features for {data_type}.")
    logging.debug(f"Feature names: {feature_names}")

    return features_df, feature_names, encoder

def train_model(X, contamination_level='auto'):
    """Trains an Isolation Forest model."""
    logging.info(f"Training Isolation Forest model with contamination={contamination_level}...")
    # Adjust parameters as needed
    model = IsolationForest(n_estimators=100, contamination=contamination_level, random_state=42, n_jobs=-1)
    try:
        model.fit(X) # Isolation Forest is unsupervised, fits only on X
        logging.info("Model training complete.")
        return model
    except ValueError as e:
        logging.error(f"Error during Isolation Forest fitting: {e}")
        logging.warning("Falling back to contamination='auto' due to fitting error.")
        # Fallback to auto if the specified level causes issues (e.g., if it's 0 or 1)
        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
        model.fit(X)
        logging.info("Model training complete with fallback contamination='auto'.")
        return model


def save_model(model, encoder, feature_names, model_dir, data_type):
    """Saves the trained model, encoder, and feature names, tagged by data_type."""
    logging.info(f"Saving {data_type} model artifacts to directory: {model_dir}")
    try:
        os.makedirs(model_dir, exist_ok=True)

        # Add data_type prefix to filenames
        model_path = os.path.join(model_dir, f'{data_type}_isolation_forest_model.joblib')
        encoder_path = os.path.join(model_dir, f'{data_type}_onehot_encoder.joblib')
        features_path = os.path.join(model_dir, f'{data_type}_feature_names.json')

        joblib.dump(model, model_path)
        joblib.dump(encoder, encoder_path)
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)

        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Encoder saved to: {encoder_path}")
        logging.info(f"Feature names saved to: {features_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        return False

def prepare_features_for_prediction(df, encoder, trained_feature_names, data_type):
    """Prepares features for new data using a loaded encoder and feature list based on data_type."""
    logging.debug(f"Preparing features for {data_type} prediction...")

    # Identify numerical and categorical features based on the encoder's input features
    categorical_features_used = encoder.feature_names_in_

    # Infer numerical features by checking trained_feature_names against encoded categorical names
    encoded_cat_feature_names_set = set(encoder.get_feature_names_out(categorical_features_used))
    numerical_features_used = [f for f in trained_feature_names if f not in encoded_cat_feature_names_set]

    # Check available columns in the input df
    available_numerical = [f for f in numerical_features_used if f in df.columns]
    available_categorical = [f for f in categorical_features_used if f in df.columns]

    missing_numerical = set(numerical_features_used) - set(available_numerical)
    missing_categorical = set(categorical_features_used) - set(available_categorical)
    if missing_numerical:
        logging.warning(f"Prediction: Missing numerical features: {missing_numerical}. Will be filled with 0.")
    if missing_categorical:
        logging.warning(f"Prediction: Missing categorical features: {missing_categorical}. Will be treated as 'missing'.")

    # Create DataFrame with expected columns, filling missing ones
    df_processed = pd.DataFrame()
    # Add numerical columns, filling missing with 0
    for col in numerical_features_used:
        df_processed[col] = df[col] if col in df.columns else 0
    # Add categorical columns, filling missing with 'missing'
    for col in categorical_features_used:
        df_processed[col] = df[col] if col in df.columns else 'missing'

    # Handle potential NaN in numerical features (redundant if filled above, but safe)
    df_numerical = df_processed[numerical_features_used].fillna(0)

    # Handle categorical features using the loaded encoder's transform method
    df_categorical = df_processed[categorical_features_used].astype(str).fillna('missing')
    try:
        encoded_cats = encoder.transform(df_categorical)
        encoded_cat_feature_names_out = encoder.get_feature_names_out(categorical_features_used)
    except Exception as e:
        logging.error(f"Error applying OneHotEncoder transform: {e}")
        return None

    # Combine numerical and encoded categorical features
    features_df = pd.concat([
        df_numerical.reset_index(drop=True),
        pd.DataFrame(encoded_cats, columns=encoded_cat_feature_names_out).reset_index(drop=True)
    ], axis=1)

    # Ensure columns match the order used during training
    try:
        features_df = features_df[trained_feature_names]
    except KeyError as e:
        logging.error(f"Prediction: Mismatch between expected features and available features: {e}")
        logging.error(f"Expected: {trained_feature_names}")
        logging.error(f"Available: {features_df.columns.tolist()}")
        # Attempt to reorder and fill missing if possible
        missing_cols = set(trained_feature_names) - set(features_df.columns)
        for c in missing_cols:
            features_df[c] = 0 # Add missing columns with 0
        try:
            features_df = features_df[trained_feature_names] # Try reordering again
            logging.warning("Reordered columns and filled missing ones with 0 for prediction.")
        except Exception as reorder_e:
            logging.error(f"Failed to reorder/fill columns for prediction: {reorder_e}")
            return None

    # Ensure all features are numeric and handle any remaining NaNs
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    logging.debug(f"Prepared prediction feature matrix with shape: {features_df.shape}")
    return features_df

# --- Prediction Function ---
def predict_anomalies(new_data_df, model_dir, data_type):
    """Loads model artifacts and predicts anomalies on new data based on data_type."""
    logging.info(f"Loading {data_type} model and making predictions...")
    try:
        # Use data_type prefix for filenames
        model_path = os.path.join(model_dir, f'{data_type}_isolation_forest_model.joblib')
        encoder_path = os.path.join(model_dir, f'{data_type}_onehot_encoder.joblib')
        features_path = os.path.join(model_dir, f'{data_type}_feature_names.json')

        if not all(os.path.exists(p) for p in [model_path, encoder_path, features_path]):
            logging.error(f"Error: One or more {data_type} model artifacts not found in {model_dir}")
            return None, None

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        with open(features_path, 'r') as f:
            trained_feature_names = json.load(f)

        # Prepare features for new data using the loaded encoder and feature list
        X_new = prepare_features_for_prediction(new_data_df, encoder, trained_feature_names, data_type)
        if X_new is None:
            logging.error("Feature preparation for prediction failed.")
            return None, None

        predictions = model.predict(X_new) # -1 for anomalies, 1 for normal
        anomaly_scores = model.decision_function(X_new)

        logging.info("Prediction complete.")
        return predictions, anomaly_scores
    except FileNotFoundError:
        logging.error(f"Error: {data_type} model artifacts not found in {model_dir}")
        return None, None
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None, None

def evaluate_model(y_true, y_pred_isoforest, anomaly_scores, data_type):
    """Calculates and logs evaluation metrics, including AUC."""
    logging.info(f"Evaluating {data_type} model performance...")

    # Convert Isolation Forest predictions (-1 anomaly, 1 normal) to (1 attack, 0 normal)
    y_pred = np.where(y_pred_isoforest == -1, 1, 0)

    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")

        # Calculate AUC using anomaly scores (lower score = more anomalous)
        # Need to check if there are both classes present for AUC calculation
        if len(np.unique(y_true)) > 1:
            try:
                # Ensure scores are inverted (lower score = more anomalous/attack)
                auc = roc_auc_score(y_true, -anomaly_scores)
                logging.info(f"AUC-ROC: {auc:.4f}")
            except ValueError as auc_e:
                logging.warning(f"Could not calculate AUC: {auc_e}")
        else:
            logging.warning("Only one class present in y_true. AUC calculation skipped.")

    except Exception as e:
        logging.error(f"Error calculating evaluation metrics: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate an Isolation Forest model on processed logs.")
    parser.add_argument('--data-path', default=DEFAULT_PROCESSED_DATA_PATH,
                        help=f"Path to the processed Parquet data file (default: {DEFAULT_PROCESSED_DATA_PATH})")
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save the trained model artifacts (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument('--test-size', type=float, default=0.3, help="Proportion of data to use for the test set.")
    parser.add_argument('--data-type', default=DEFAULT_DATA_TYPE, choices=['cloudtrail', 'vpc'],
                        help=f"Type of log data to process ('cloudtrail' or 'vpc', default: {DEFAULT_DATA_TYPE})")

    args = parser.parse_args()

    # 1. Load Data
    df = load_data(args.data_path, args.data_type)

    if df is not None:
        # Define target variable
        y = df['is_simulated_attack']

        # 2. Prepare Features (on the entire dataset for consistent encoding)
        X_prepared, feature_names, encoder = prepare_features(df.drop(columns=['is_simulated_attack']), args.data_type)

        if X_prepared is not None and feature_names is not None and encoder is not None:

            # 3. Split Data into Training and Testing sets
            logging.info(f"Splitting data into train/test sets (Test size: {args.test_size}) for {args.data_type}")
            try:
                # Check if there are enough samples for stratification
                if y.nunique() < 2:
                     logging.warning("Only one class present in labels. Cannot stratify. Performing regular split.")
                     stratify_param = None
                else:
                     stratify_param = y

                X_train, X_test, y_train, y_test = train_test_split(
                    X_prepared, y, test_size=args.test_size, random_state=42, stratify=stratify_param
                )
                logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
                logging.info(f"Attack ratio in training set: {y_train.mean():.4f}")
                logging.info(f"Attack ratio in test set: {y_test.mean():.4f}")
            except ValueError as e:
                 logging.error(f"Error during train_test_split (possibly too few samples for a class): {e}. Performing regular split.")
                 try:
                     X_train, X_test, y_train, y_test = train_test_split(
                         X_prepared, y, test_size=args.test_size, random_state=42
                     )
                     logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
                 except Exception as split_err:
                     logging.error(f"Failed to split data even without stratification: {split_err}")
                     X_train, X_test, y_train, y_test = None, None, None, None
            except Exception as e:
                logging.error(f"Error splitting data: {e}. Check if labels exist and data is sufficient.")
                X_train, X_test, y_train, y_test = None, None, None, None

            if X_train is not None and X_test is not None:
                # 4. Train Model (on training data only)
                # Estimate contamination based on training data labels
                estimated_contamination = y_train.mean()
                # Ensure contamination is within the valid range (0, 0.5]
                if estimated_contamination <= 0 or estimated_contamination > 0.5:
                    logging.warning(f"Estimated contamination {estimated_contamination:.4f} is outside (0, 0.5]. Using 'auto'.")
                    contamination_level = 'auto'
                else:
                    contamination_level = estimated_contamination
                    logging.info(f"Using estimated contamination: {contamination_level:.4f}")

                # Note: Isolation Forest is unsupervised, doesn't use y_train for fitting itself
                model = train_model(X_train, contamination_level=contamination_level)

                # 5. Save Model (trained on X_train)
                save_model(model, encoder, feature_names, args.model_dir, args.data_type)

                # 6. Predict on Test Set
                logging.info("Making predictions on the test set...")
                test_predictions_isoforest, test_anomaly_scores = model.predict(X_test), model.decision_function(X_test)

                # 7. Evaluate Model
                evaluate_model(y_test, test_predictions_isoforest, test_anomaly_scores, args.data_type)

                # --- Optional: Example Prediction Call on new data (if needed) ---
                # logging.info(f"--- Running Example Prediction on {args.data_type} Test Data Head ---")
                # sample_data_for_prediction = df.loc[y_test.index].head() # Get original data for test indices
                # predictions, scores = predict_anomalies(sample_data_for_prediction, args.model_dir, args.data_type)
                # if predictions is not None:
                #     logging.info(f"Sample Predictions (First {len(predictions)}): {predictions}")
                #     logging.info(f"Sample Anomaly Scores (First {len(scores)}): {scores}")
            else:
                logging.error("Data splitting failed. Cannot proceed with training and evaluation.")
        else:
            logging.error("Feature preparation failed. Exiting.")
    else:
        logging.error("Data loading failed. Exiting.")

    logging.info("ML engine script finished.")