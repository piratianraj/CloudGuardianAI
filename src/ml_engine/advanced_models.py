# Implementation of advanced ML models for anomaly detection on processed CloudTrail or VPC Flow logs
# Includes: Autoencoder, One-Class SVM, and XGBoost/Random Forest

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# For Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [ADVANCED_ML] - %(message)s')

# ===== AUTOENCODER MODEL =====

def train_autoencoder(X_train, X_test=None, epochs=50, batch_size=32, validation_split=0.1):
    """
    Trains an Autoencoder neural network for anomaly detection.
    
    Args:
        X_train: Training data features
        X_test: Optional test data for validation during training
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Proportion of training data to use for validation if X_test is None
        
    Returns:
        model: Trained autoencoder model
        threshold: Reconstruction error threshold for anomaly detection
        scaler: Fitted StandardScaler for feature normalization
    """
    logging.info("Training Autoencoder model...")
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        validation_data = (X_test_scaled, X_test_scaled)
        validation_split = None
    else:
        validation_data = None
    
    # Get input dimensions
    input_dim = X_train.shape[1]
    
    # Define model architecture
    # The architecture can be adjusted based on the complexity of the data
    encoding_dim = min(64, input_dim)  # Encoding dimension
    hidden_dim = min(32, encoding_dim)  # Hidden dimension
    
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)  # Add dropout for regularization
    encoder = Dense(hidden_dim, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(encoding_dim, activation='relu')(encoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Compile model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    try:
        history = autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate reconstruction error on training data
        reconstructions = autoencoder.predict(X_train_scaled)
        train_mse = np.mean(np.power(X_train_scaled - reconstructions, 2), axis=1)
        
        # Set threshold for anomaly detection (e.g., mean + 2*std)
        threshold = np.mean(train_mse) + 2 * np.std(train_mse)
        logging.info(f"Autoencoder training complete. Reconstruction error threshold: {threshold:.6f}")
        
        return autoencoder, threshold, scaler
        
    except Exception as e:
        logging.error(f"Error during Autoencoder training: {e}")
        return None, None, None

def predict_autoencoder(model, threshold, scaler, X_new):
    """
    Predicts anomalies using the trained Autoencoder model.
    
    Args:
        model: Trained Autoencoder model
        threshold: Reconstruction error threshold for anomaly detection
        scaler: Fitted StandardScaler
        X_new: New data for prediction
        
    Returns:
        predictions: Binary predictions (1 for anomaly, 0 for normal)
        anomaly_scores: Reconstruction errors (higher = more anomalous)
    """
    try:
        # Scale the data
        X_scaled = scaler.transform(X_new)
        
        # Get reconstructions
        reconstructions = model.predict(X_scaled)
        
        # Calculate reconstruction error (MSE)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        
        # Classify as anomaly if error > threshold
        predictions = np.where(mse > threshold, 1, 0)
        
        return predictions, mse
    except Exception as e:
        logging.error(f"Error during Autoencoder prediction: {e}")
        return None, None

def save_autoencoder(model, threshold, scaler, model_dir, data_type):
    """
    Saves the trained Autoencoder model and associated artifacts.
    
    Args:
        model: Trained Autoencoder model
        threshold: Reconstruction error threshold
        scaler: Fitted StandardScaler
        model_dir: Directory to save model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
    """
    logging.info(f"Saving {data_type} Autoencoder model artifacts to directory: {model_dir}")
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f'{data_type}_autoencoder_model.h5')
        model.save(model_path)
        
        # Save threshold and scaler
        threshold_path = os.path.join(model_dir, f'{data_type}_autoencoder_threshold.joblib')
        scaler_path = os.path.join(model_dir, f'{data_type}_autoencoder_scaler.joblib')
        
        joblib.dump(threshold, threshold_path)
        joblib.dump(scaler, scaler_path)
        
        logging.info(f"Autoencoder model saved to: {model_path}")
        logging.info(f"Threshold saved to: {threshold_path}")
        logging.info(f"Scaler saved to: {scaler_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving Autoencoder model artifacts: {e}")
        return False

def load_autoencoder(model_dir, data_type):
    """
    Loads a trained Autoencoder model and associated artifacts.
    
    Args:
        model_dir: Directory containing model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
        
    Returns:
        model: Loaded Autoencoder model
        threshold: Reconstruction error threshold
        scaler: Fitted StandardScaler
    """
    logging.info(f"Loading {data_type} Autoencoder model artifacts from directory: {model_dir}")
    try:
        model_path = os.path.join(model_dir, f'{data_type}_autoencoder_model.h5')
        threshold_path = os.path.join(model_dir, f'{data_type}_autoencoder_threshold.joblib')
        scaler_path = os.path.join(model_dir, f'{data_type}_autoencoder_scaler.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, threshold_path, scaler_path]):
            logging.error(f"Error: One or more {data_type} Autoencoder model artifacts not found")
            return None, None, None
        
        model = load_model(model_path)
        threshold = joblib.load(threshold_path)
        scaler = joblib.load(scaler_path)
        
        logging.info(f"Autoencoder model loaded from: {model_path}")
        
        return model, threshold, scaler
    except Exception as e:
        logging.error(f"Error loading Autoencoder model artifacts: {e}")
        return None, None, None

# ===== ONE-CLASS SVM MODEL =====

def train_ocsvm(X_train, nu=0.1, kernel='rbf'):
    """
    Trains a One-Class SVM model for anomaly detection.
    
    Args:
        X_train: Training data features
        nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
        kernel: Kernel type to be used in the algorithm
        
    Returns:
        model: Trained One-Class SVM model
        scaler: Fitted StandardScaler
    """
    logging.info(f"Training One-Class SVM model with nu={nu}, kernel={kernel}...")
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    try:
        model = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')
        model.fit(X_train_scaled)
        logging.info("One-Class SVM training complete.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error during One-Class SVM training: {e}")
        return None, None

def predict_ocsvm(model, scaler, X_new):
    """
    Predicts anomalies using the trained One-Class SVM model.
    
    Args:
        model: Trained One-Class SVM model
        scaler: Fitted StandardScaler
        X_new: New data for prediction
        
    Returns:
        predictions: Binary predictions (1 for anomaly, 0 for normal)
        anomaly_scores: Decision function values (lower = more anomalous)
    """
    try:
        # Scale the data
        X_scaled = scaler.transform(X_new)
        
        # Get predictions and decision function values
        # One-Class SVM returns: 1 for inliers, -1 for outliers
        raw_predictions = model.predict(X_scaled)
        decision_values = model.decision_function(X_scaled)
        
        # Convert to our convention: 1 for anomaly (attack), 0 for normal
        predictions = np.where(raw_predictions == -1, 1, 0)
        
        return predictions, decision_values
    except Exception as e:
        logging.error(f"Error during One-Class SVM prediction: {e}")
        return None, None

def save_ocsvm(model, scaler, model_dir, data_type):
    """
    Saves the trained One-Class SVM model and scaler.
    
    Args:
        model: Trained One-Class SVM model
        scaler: Fitted StandardScaler
        model_dir: Directory to save model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
    """
    logging.info(f"Saving {data_type} One-Class SVM model artifacts to directory: {model_dir}")
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'{data_type}_ocsvm_model.joblib')
        scaler_path = os.path.join(model_dir, f'{data_type}_ocsvm_scaler.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logging.info(f"One-Class SVM model saved to: {model_path}")
        logging.info(f"Scaler saved to: {scaler_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving One-Class SVM model artifacts: {e}")
        return False

def load_ocsvm(model_dir, data_type):
    """
    Loads a trained One-Class SVM model and scaler.
    
    Args:
        model_dir: Directory containing model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
        
    Returns:
        model: Loaded One-Class SVM model
        scaler: Fitted StandardScaler
    """
    logging.info(f"Loading {data_type} One-Class SVM model artifacts from directory: {model_dir}")
    try:
        model_path = os.path.join(model_dir, f'{data_type}_ocsvm_model.joblib')
        scaler_path = os.path.join(model_dir, f'{data_type}_ocsvm_scaler.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path]):
            logging.error(f"Error: One or more {data_type} One-Class SVM model artifacts not found")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logging.info(f"One-Class SVM model loaded from: {model_path}")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading One-Class SVM model artifacts: {e}")
        return None, None

# ===== XGBOOST/RANDOM FOREST MODEL =====

def train_supervised_model(X_train, y_train, model_type='random_forest', class_weight='balanced'):
    """
    Trains a supervised model (Random Forest or XGBoost) for classification.
    
    Args:
        X_train: Training data features
        y_train: Training data labels
        model_type: Type of model to train ('random_forest' or 'xgboost')
        class_weight: Weight for handling class imbalance
        
    Returns:
        model: Trained supervised model
        scaler: Fitted StandardScaler
    """
    logging.info(f"Training {model_type.replace('_', ' ').title()} model...")
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    try:
        if model_type == 'random_forest':
            # Random Forest with class weights to handle imbalance
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            # Using Gradient Boosting as a proxy for XGBoost (which would require additional package)
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            # If using actual XGBoost, you would handle class weights differently
            # For example: scale_pos_weight parameter for binary classification
        else:
            logging.error(f"Unknown model type: {model_type}")
            return None, None
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        logging.info(f"{model_type.replace('_', ' ').title()} training complete.")
        
        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = min(10, len(indices))
            
            logging.info("Top feature importances:")
            for i in range(top_n):
                logging.info(f"  {i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error during {model_type} training: {e}")
        return None, None

def predict_supervised(model, scaler, X_new):
    """
    Predicts using the trained supervised model.
    
    Args:
        model: Trained supervised model
        scaler: Fitted StandardScaler
        X_new: New data for prediction
        
    Returns:
        predictions: Binary predictions (1 for attack, 0 for normal)
        probabilities: Probability of the positive class (attack)
    """
    try:
        # Scale the data
        X_scaled = scaler.transform(X_new)
        
        # Get predictions and probabilities
        predictions = model.predict(X_scaled)
        
        # Get probabilities for the positive class (attack)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            # If predict_proba is not available, use decision function
            probabilities = model.decision_function(X_scaled)
        
        return predictions, probabilities
    except Exception as e:
        logging.error(f"Error during supervised model prediction: {e}")
        return None, None

def save_supervised_model(model, scaler, model_dir, data_type, model_type):
    """
    Saves the trained supervised model and scaler.
    
    Args:
        model: Trained supervised model
        scaler: Fitted StandardScaler
        model_dir: Directory to save model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
        model_type: Type of model ('random_forest' or 'xgboost')
    """
    logging.info(f"Saving {data_type} {model_type} model artifacts to directory: {model_dir}")
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'{data_type}_{model_type}_model.joblib')
        scaler_path = os.path.join(model_dir, f'{data_type}_{model_type}_scaler.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logging.info(f"{model_type.replace('_', ' ').title()} model saved to: {model_path}")
        logging.info(f"Scaler saved to: {scaler_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving {model_type} model artifacts: {e}")
        return False

def load_supervised_model(model_dir, data_type, model_type):
    """
    Loads a trained supervised model and scaler.
    
    Args:
        model_dir: Directory containing model artifacts
        data_type: Type of data ('cloudtrail' or 'vpc')
        model_type: Type of model ('random_forest' or 'xgboost')
        
    Returns:
        model: Loaded supervised model
        scaler: Fitted StandardScaler
    """
    logging.info(f"Loading {data_type} {model_type} model artifacts from directory: {model_dir}")
    try:
        model_path = os.path.join(model_dir, f'{data_type}_{model_type}_model.joblib')
        scaler_path = os.path.join(model_dir, f'{data_type}_{model_type}_scaler.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path]):
            logging.error(f"Error: One or more {data_type} {model_type} model artifacts not found")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logging.info(f"{model_type.replace('_', ' ').title()} model loaded from: {model_path}")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading {model_type} model artifacts: {e}")
        return None, None

# ===== EVALUATION FUNCTIONS =====

def evaluate_model_performance(y_true, y_pred, anomaly_scores=None, model_name=""):
    """
    Evaluates model performance using various metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        anomaly_scores: Anomaly scores or probabilities (optional)
        model_name: Name of the model being evaluated
    """
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        logging.info(f"{model_name} Evaluation Results:")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")
        
        # Calculate AUC if scores are provided and there are both classes present
        if anomaly_scores is not None and len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, anomaly_scores)
                logging.info(f"AUC-ROC: {auc:.4f}")
            except ValueError as auc_e:
                logging.warning(f"Could not calculate AUC: {auc_e}")
        elif len(np.unique(y_true)) <= 1:
            logging.warning("Only one class present in y_true. AUC calculation skipped.")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'auc': auc if 'auc' in locals() else None
        }
    except Exception as e:
        logging.error(f"Error calculating evaluation metrics: {e}")
        return None

# ===== HELPER FUNCTIONS =====

def compare_models(models_results, title="Model Comparison"):
    """
    Compares multiple models based on their evaluation metrics.
    
    Args:
        models_results: Dictionary of model names and their evaluation results
        title: Title for the comparison plot
    """
    try:
        model_names = list(models_results.keys())
        metrics = ['precision', 'recall', 'f1', 'auc']
        
        # Extract metrics for each model
        data = {}
        for metric in metrics:
            data[metric] = [models_results[model].get(metric, 0) for model in model_names]
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.2
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            if any(v is not None for v in data[metric]):
                # Replace None with 0 for plotting
                values = [v if v is not None else 0 for v in data[metric]]
                ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        # Add labels and legend
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        logging.info("Model comparison plot saved to 'model_comparison.png'")
        
        # Print comparison table
        logging.info("\nModel Comparison Table:")
        header = "Model\t\t" + "\t".join([m.capitalize() for m in metrics])
        logging.info(header)
        logging.info("-" * len(header) * 2)
        
        for i, model in enumerate(model_names):
            row = f"{model}\t\t"
            for metric in metrics:
                value = data[metric][i]
                row += f"{value:.4f}\t" if value is not None else "N/A\t\t"
            logging.info(row)
        
    except Exception as e:
        logging.error(f"Error creating model comparison: {e}")