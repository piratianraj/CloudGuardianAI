# Implementation of sequence-based ML models for analyzing API call patterns
# Includes: LSTM and Transformer models for detecting unusual API call sequences

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# For sequence models
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [SEQUENCE_ML] - %(message)s')

# ===== DATA PREPARATION =====

def prepare_sequence_data(df, sequence_length=10, step=1, user_col='user_name', 
                         event_col='eventName', timestamp_col='eventTime'):
    """
    Prepare sequence data for LSTM model from CloudTrail logs.
    
    Args:
        df: DataFrame containing CloudTrail logs
        sequence_length: Length of API call sequences to create
        step: Step size for sliding window
        user_col: Column containing user identifiers
        event_col: Column containing API call names
        timestamp_col: Column containing timestamps
        
    Returns:
        sequences: Array of API call sequences
        next_calls: Array of next API calls (for supervised learning)
        label_encoder: Fitted LabelEncoder for API call names
    """
    logging.info(f"Preparing sequence data with sequence length {sequence_length}...")
    
    # Ensure timestamp is datetime
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
    
    # Encode API call names
    label_encoder = LabelEncoder()
    df['event_encoded'] = label_encoder.fit_transform(df[event_col])
    
    sequences = []
    next_calls = []
    
    # Group by user
    for user, user_df in df.groupby(user_col):
        # Get sequence of API calls
        api_calls = user_df['event_encoded'].values
        
        # Create sequences
        for i in range(0, len(api_calls) - sequence_length, step):
            seq = api_calls[i:i+sequence_length]
            next_call = api_calls[i+sequence_length]
            sequences.append(seq)
            next_calls.append(next_call)
    
    logging.info(f"Created {len(sequences)} sequences from {len(df[user_col].unique())} users")
    return np.array(sequences), np.array(next_calls), label_encoder

# ===== LSTM MODEL =====

def train_lstm_sequence_model(X_train, y_train, vocab_size, embedding_dim=64, 
                             lstm_units=128, epochs=50, batch_size=64):
    """
    Train LSTM model for API call sequence prediction.
    
    Args:
        X_train: Training sequences
        y_train: Target next API calls
        vocab_size: Size of API call vocabulary
        embedding_dim: Dimension of embedding layer
        lstm_units: Number of LSTM units
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model: Trained LSTM model
        history: Training history
    """
    logging.info("Training LSTM sequence model...")
    
    # Build model
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=X_train.shape[1]),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(lstm_units)),
        Dropout(0.3),
        Dense(vocab_size, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    logging.info(f"LSTM model training completed with final accuracy: {history.history['accuracy'][-1]:.4f}")
    return model, history

# ===== ANOMALY DETECTION =====

def detect_sequence_anomalies(model, sequences, actual_next_calls, threshold=0.01):
    """
    Detect anomalies in API call sequences using the trained LSTM model.
    
    Args:
        model: Trained LSTM model
        sequences: API call sequences to evaluate
        actual_next_calls: Actual next API calls
        threshold: Probability threshold for anomaly detection
        
    Returns:
        anomalies: Boolean array indicating anomalous sequences
        anomaly_scores: Array of anomaly scores
    """
    logging.info(f"Detecting sequence anomalies with threshold {threshold}...")
    
    # Predict next API call probabilities
    predictions = model.predict(sequences)
    
    # Get the probability assigned to the actual next call
    anomaly_scores = []
    for i, actual in enumerate(actual_next_calls):
        prob = predictions[i, actual]
        anomaly_scores.append(1 - prob)  # Higher score = more anomalous
    
    # Flag sequences with low probability as anomalies
    anomaly_scores = np.array(anomaly_scores)
    anomalies = anomaly_scores > threshold
    
    anomaly_count = np.sum(anomalies)
    logging.info(f"Detected {anomaly_count} anomalous sequences ({anomaly_count/len(sequences):.2%})")
    
    return anomalies, anomaly_scores

# ===== MODEL SAVING/LOADING =====

def save_sequence_model(model, label_encoder, threshold, output_dir, prefix='cloudtrail'):
    """
    Save sequence model and associated artifacts.
    
    Args:
        model: Trained sequence model
        label_encoder: Fitted LabelEncoder
        threshold: Anomaly detection threshold
        output_dir: Directory to save model artifacts
        prefix: Prefix for saved files
        
    Returns:
        None
    """
    logging.info(f"Saving sequence model to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{prefix}_lstm_model.h5")
    model.save(model_path)
    
    # Save label encoder
    encoder_path = os.path.join(output_dir, f"{prefix}_lstm_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    
    # Save threshold
    threshold_path = os.path.join(output_dir, f"{prefix}_lstm_threshold.joblib")
    joblib.dump(threshold, threshold_path)
    
    logging.info("Sequence model and artifacts saved successfully")

def load_sequence_model(model_dir, prefix='cloudtrail'):
    """
    Load sequence model and associated artifacts.
    
    Args:
        model_dir: Directory containing model artifacts
        prefix: Prefix for saved files
        
    Returns:
        model: Loaded sequence model
        label_encoder: Loaded LabelEncoder
        threshold: Loaded anomaly detection threshold
    """
    logging.info(f"Loading sequence model from {model_dir}...")
    
    # Load model
    model_path = os.path.join(model_dir, f"{prefix}_lstm_model.h5")
    model = load_model(model_path)
    
    # Load label encoder
    encoder_path = os.path.join(model_dir, f"{prefix}_lstm_encoder.joblib")
    label_encoder = joblib.load(encoder_path)
    
    # Load threshold
    threshold_path = os.path.join(model_dir, f"{prefix}_lstm_threshold.joblib")
    threshold = joblib.load(threshold_path)
    
    logging.info("Sequence model and artifacts loaded successfully")
    return model, label_encoder, threshold

# ===== PREDICTION FUNCTION =====

def predict_sequence_anomalies(df, model_dir, prefix='cloudtrail', sequence_length=10,
                             user_col='user_name', event_col='eventName', timestamp_col='eventTime'):
    """
    Predict sequence anomalies on new data.
    
    Args:
        df: DataFrame containing CloudTrail logs
        model_dir: Directory containing model artifacts
        prefix: Prefix for saved files
        sequence_length: Length of API call sequences
        user_col: Column containing user identifiers
        event_col: Column containing API call names
        timestamp_col: Column containing timestamps
        
    Returns:
        df_results: DataFrame with anomaly scores and flags
    """
    logging.info("Predicting sequence anomalies on new data...")
    
    # Load model and artifacts
    model, label_encoder, threshold = load_sequence_model(model_dir, prefix)
    
    # Prepare sequences
    # First encode API calls using the loaded encoder
    df['event_encoded'] = df[event_col].map(lambda x: label_encoder.transform([x])[0] 
                                         if x in label_encoder.classes_ else -1)
    
    # Drop rows with unknown API calls
    df = df[df['event_encoded'] != -1].copy()
    
    # Create result DataFrame
    df_results = df.copy()
    df_results['sequence_anomaly_score'] = 0.0
    df_results['is_sequence_anomaly'] = False
    
    # Group by user
    for user, user_df in df.groupby(user_col):
        # Get sequence of API calls
        api_calls = user_df['event_encoded'].values
        
        # Skip if not enough calls
        if len(api_calls) <= sequence_length:
            continue
        
        # Create sequences
        sequences = []
        sequence_indices = []
        
        for i in range(0, len(api_calls) - sequence_length):
            seq = api_calls[i:i+sequence_length]
            next_call = api_calls[i+sequence_length]
            sequences.append(seq)
            sequence_indices.append(i+sequence_length)  # Index of the next call
        
        if not sequences:
            continue
        
        # Convert to numpy arrays
        sequences = np.array(sequences)
        next_calls = api_calls[sequence_indices]
        
        # Detect anomalies
        anomalies, anomaly_scores = detect_sequence_anomalies(model, sequences, next_calls, threshold)
        
        # Add results to DataFrame
        for i, idx in enumerate(sequence_indices):
            df_results.iloc[user_df.index[idx], df_results.columns.get_loc('sequence_anomaly_score')] = anomaly_scores[i]
            df_results.iloc[user_df.index[idx], df_results.columns.get_loc('is_sequence_anomaly')] = anomalies[i]
    
    anomaly_count = df_results['is_sequence_anomaly'].sum()
    logging.info(f"Detected {anomaly_count} sequence anomalies in {len(df_results)} events")
    
    return df_results