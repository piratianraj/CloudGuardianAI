# Streamlit visualization dashboard for Cloud Threat Detector

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import logging

# --- Add project root to path to import ml_engine --- #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

# --- Import prediction functions --- #
# Ensure ml_engine and its dependencies (joblib, sklearn, etc.) are installed
try:
    # Import from original isolation forest implementation
    from ml_engine.train_predict import predict_anomalies as predict_isolation_forest, load_data as load_processed_data
    
    # Import from advanced models implementation
    from ml_engine.advanced_models import (
        load_autoencoder, predict_autoencoder,
        load_ocsvm, predict_ocsvm,
        load_supervised_model, predict_supervised
    )
    
    # Flag to indicate if advanced models are available
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing ML engine: {e}. Make sure dependencies are installed and paths are correct.")
    ADVANCED_MODELS_AVAILABLE = False
    st.warning("Advanced models not available. Only Isolation Forest will be used.")
except Exception as e:
    st.error(f"An unexpected error occurred during import: {e}")
    ADVANCED_MODELS_AVAILABLE = False
    st.stop()

# Configure basic logging for the dashboard
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [DASHBOARD] - %(message)s')

# --- Constants --- #
DEFAULT_PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_cloudtrail_logs.parquet')
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Page Configuration --- #
st.set_page_config(
    page_title="Cloud Threat Detector Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("☁️ AI-Powered Cloud Threat Detector")

st.sidebar.header("Configuration")
# Add configuration options
data_path_input = st.sidebar.text_input("Processed Data Path", DEFAULT_PROCESSED_DATA_PATH)
model_dir_input = st.sidebar.text_input("Model Directory", DEFAULT_MODEL_DIR)

# Data type selection
data_type = st.sidebar.selectbox(
    "Data Type",
    options=["cloudtrail", "vpc"],
    index=0,
    help="Select the type of log data to analyze"
)

# Model selection
model_options = ["Isolation Forest"]
if ADVANCED_MODELS_AVAILABLE:
    model_options.extend(["Autoencoder", "One-Class SVM", "Random Forest", "XGBoost"])
    
selected_model = st.sidebar.selectbox(
    "Model",
    options=model_options,
    index=0,
    help="Select the model to use for anomaly detection"
)

# Threshold for anomaly detection (model-specific)
if selected_model == "Isolation Forest":
    anomaly_threshold = st.sidebar.slider("Anomaly Score Threshold", 
                                        min_value=-1.0, max_value=1.0, 
                                        value=-0.1, step=0.05,
                                        help="Lower scores indicate more anomalous events")
elif selected_model == "Autoencoder":
    anomaly_threshold = st.sidebar.slider("Reconstruction Error Threshold", 
                                        min_value=0.0, max_value=1.0, 
                                        value=0.3, step=0.01,
                                        help="Higher errors indicate more anomalous events")
elif selected_model == "One-Class SVM":
    anomaly_threshold = st.sidebar.slider("Decision Function Threshold", 
                                        min_value=-1.0, max_value=1.0, 
                                        value=0.0, step=0.05,
                                        help="Lower values indicate more anomalous events")
else:  # Supervised models (Random Forest, XGBoost)
    anomaly_threshold = st.sidebar.slider("Probability Threshold", 
                                        min_value=0.0, max_value=1.0, 
                                        value=0.5, step=0.05,
                                        help="Higher probabilities indicate more likely attacks")

# --- Data Loading and Prediction --- #
@st.cache_data # Cache data loading and prediction
def load_and_predict(data_path, model_dir, data_type, model_name):
    logging.info(f"Attempting to load {data_type} data and run predictions using {model_name}...")
    df = load_processed_data(data_path, data_type)
    if df is None or df.empty:
        logging.error("Failed to load data or data is empty.")
        return pd.DataFrame() # Return empty DataFrame on failure

    # Different prediction logic based on selected model
    if model_name == "Isolation Forest":
        predictions, scores = predict_isolation_forest(df, model_dir, data_type)
        if predictions is not None:
            # Convert Isolation Forest predictions (-1 anomaly, 1 normal) to (1 attack, 0 normal)
            df['anomaly_prediction'] = np.where(predictions == -1, 1, 0)
            df['anomaly_score'] = -scores  # Invert scores so higher = more anomalous
            logging.info("Isolation Forest predictions added to DataFrame.")
    
    elif model_name == "Autoencoder" and ADVANCED_MODELS_AVAILABLE:
        try:
            model, threshold, scaler = load_autoencoder(model_dir, data_type)
            if model is not None:
                # Prepare features (similar to what's done in train_predict.py)
                from ml_engine.train_predict import prepare_features
                X_prepared, _, _ = prepare_features(df, data_type)
                if X_prepared is not None:
                    predictions, scores = predict_autoencoder(model, threshold, scaler, X_prepared)
                    df['anomaly_prediction'] = predictions  # 1 for anomaly, 0 for normal
                    df['anomaly_score'] = scores  # Higher = more anomalous
                    logging.info("Autoencoder predictions added to DataFrame.")
        except Exception as e:
            logging.error(f"Error with Autoencoder prediction: {e}")
    
    elif model_name == "One-Class SVM" and ADVANCED_MODELS_AVAILABLE:
        try:
            model, scaler = load_ocsvm(model_dir, data_type)
            if model is not None:
                from ml_engine.train_predict import prepare_features
                X_prepared, _, _ = prepare_features(df, data_type)
                if X_prepared is not None:
                    predictions, scores = predict_ocsvm(model, scaler, X_prepared)
                    df['anomaly_prediction'] = predictions  # 1 for anomaly, 0 for normal
                    df['anomaly_score'] = -scores  # Invert so higher = more anomalous
                    logging.info("One-Class SVM predictions added to DataFrame.")
        except Exception as e:
            logging.error(f"Error with One-Class SVM prediction: {e}")
    
    elif model_name in ["Random Forest", "XGBoost"] and ADVANCED_MODELS_AVAILABLE:
        try:
            model_type = 'random_forest' if model_name == "Random Forest" else 'xgboost'
            model, scaler = load_supervised_model(model_dir, data_type, model_type)
            if model is not None:
                from ml_engine.train_predict import prepare_features
                X_prepared, _, _ = prepare_features(df, data_type)
                if X_prepared is not None:
                    predictions, probabilities = predict_supervised(model, scaler, X_prepared)
                    df['anomaly_prediction'] = predictions  # 1 for attack, 0 for normal
                    df['anomaly_score'] = probabilities  # Higher = more likely attack
                    logging.info(f"{model_name} predictions added to DataFrame.")
        except Exception as e:
            logging.error(f"Error with {model_name} prediction: {e}")
    
    else:
        logging.error(f"Unknown or unavailable model: {model_name}")
        # Return original data without predictions if prediction fails
        if 'anomaly_prediction' not in df.columns:
            df['anomaly_prediction'] = 0
            df['anomaly_score'] = 0
    
    return df

# Load data and get predictions
df_processed = load_and_predict(data_path_input, model_dir_input, data_type, selected_model)

if df_processed.empty:
    st.error(f"Could not load or process data from {data_path_input}. Please check the path and ensure the ML engine ran successfully.")
    st.stop()

# --- Dashboard Sections --- #
st.header("🚨 Detected Anomalies")

# Filter alerts based on prediction and score threshold
# For Isolation Forest, prediction is 1 (was converted from -1) and higher scores are more anomalous
# For Autoencoder, prediction is 1 and higher scores (reconstruction errors) are more anomalous
# For One-Class SVM, prediction is 1 and higher scores are more anomalous (after inversion)
# For supervised models, prediction is 1 and higher scores (probabilities) indicate attacks

# Check if prediction columns exist before filtering
if 'anomaly_prediction' not in df_processed.columns:
    logging.warning(f"'anomaly_prediction' column not found in data. Model prediction may have failed.")
    df_processed['anomaly_prediction'] = 0  # Add default column
    df_processed['anomaly_score'] = 0  # Add default column
    anomalies_df = pd.DataFrame(columns=df_processed.columns)  # Empty dataframe with same columns
else:
    if selected_model == "Isolation Forest":
        anomalies_df = df_processed[
            (df_processed['anomaly_prediction'] == 1) &
            (df_processed['anomaly_score'] > anomaly_threshold) # Higher scores (after inversion) are more anomalous
        ].copy()
    elif selected_model == "Autoencoder":
        anomalies_df = df_processed[
            (df_processed['anomaly_prediction'] == 1) &
            (df_processed['anomaly_score'] > anomaly_threshold) # Higher reconstruction errors are more anomalous
        ].copy()
    elif selected_model == "One-Class SVM":
        anomalies_df = df_processed[
            (df_processed['anomaly_prediction'] == 1) &
            (df_processed['anomaly_score'] > anomaly_threshold) # Higher scores (after inversion) are more anomalous
        ].copy()
    else:  # Supervised models (Random Forest, XGBoost)
        anomalies_df = df_processed[
            (df_processed['anomaly_score'] > anomaly_threshold) # Higher probabilities indicate attacks
        ].copy()

if not anomalies_df.empty:
    # Select and rename columns for better readability
    display_columns = {
        'eventTime': 'Timestamp',
        'eventName': 'Event Name',
        'user_name': 'User/Role',
        'source_ip': 'Source IP',
        'awsRegion': 'Region',
        'anomaly_score': 'Anomaly Score'
        # Add other relevant columns like 'requestParameters' if needed
    }
    # Filter columns that actually exist in the dataframe
    existing_display_columns = {k: v for k, v in display_columns.items() if k in anomalies_df.columns}
    anomalies_display = anomalies_df[list(existing_display_columns.keys())].rename(columns=existing_display_columns)

    st.dataframe(anomalies_display.sort_values(by='Anomaly Score', ascending=True), use_container_width=True)
    st.info(f"Showing {len(anomalies_display)} events detected as anomalies with score < {anomaly_threshold:.2f}")
else:
    st.success(f"No anomalies detected with score < {anomaly_threshold:.2f}")

st.header("📊 Anomaly Score Distribution Over Time")

# Create a time series plot of anomaly scores
if not df_processed.empty and 'eventTime' in df_processed.columns and 'anomaly_score' in df_processed.columns:
    # Ensure eventTime is datetime
    df_processed['eventTime'] = pd.to_datetime(df_processed['eventTime'])
    
    # Set title and legend based on model
    if selected_model == "Isolation Forest":
        title = 'Anomaly Score Trend (Higher scores are more anomalous)'
        legend_title = 'Prediction (1=Anomaly)'
    elif selected_model == "Autoencoder":
        title = 'Reconstruction Error Trend (Higher errors are more anomalous)'
        legend_title = 'Prediction (1=Anomaly)'
    elif selected_model == "One-Class SVM":
        title = 'Decision Function Trend (Higher values are more anomalous)'
        legend_title = 'Prediction (1=Anomaly)'
    else:  # Supervised models
        title = 'Attack Probability Trend (Higher probabilities indicate attacks)'
        legend_title = 'Prediction (1=Attack)'
    
    # Ensure anomaly_prediction column exists for coloring
    if 'anomaly_prediction' not in df_processed.columns:
        df_processed['anomaly_prediction'] = 0  # Add default column if missing

    fig = px.scatter(df_processed.sort_values(by='eventTime'),
                     x='eventTime',
                     y='anomaly_score',
                     title=title,
                     color='anomaly_prediction', # Color points by prediction (0 or 1)
                     color_discrete_map={0: 'blue', 1: 'red'},
                     hover_data=['eventName', 'user_name', 'source_ip'] if 'eventName' in df_processed.columns else None)

    fig.update_layout(xaxis_title='Timestamp',
                      yaxis_title='Score',
                      legend_title=legend_title)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Anomaly scores or timestamps not available for plotting.")

# --- Model Performance Metrics Section ---
st.header("📈 Model Performance Metrics")

# Display metrics if available in the data
if 'is_simulated_attack' in df_processed.columns:
    # Check if anomaly_prediction column exists
    if 'anomaly_prediction' not in df_processed.columns:
        st.warning("'anomaly_prediction' column not found. Model prediction may have failed. Cannot calculate performance metrics.")
    else:
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        import numpy as np
        
        y_true = df_processed['is_simulated_attack'].astype(int)
        y_pred = df_processed['anomaly_prediction'].astype(int)
        
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            # Create metrics columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{precision:.4f}")
            col2.metric("Recall", f"{recall:.4f}")
            col3.metric("F1-Score", f"{f1:.4f}")
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(
                cm, 
                columns=["Predicted Normal", "Predicted Anomaly"],
                index=["Actual Normal", "Actual Anomaly"]
            )
            st.dataframe(cm_df)
            
            # Calculate and display additional metrics
            tn, fp, fn, tp = cm.ravel()
            st.text(f"True Negatives: {tn}  |  False Positives: {fp}")
            st.text(f"False Negatives: {fn}  |  True Positives: {tp}")
            
        except Exception as e:
            st.warning(f"Could not calculate metrics: {e}")
else:
    st.info("Ground truth labels ('is_simulated_attack') not available for performance evaluation.")

# --- Entity Relationship Graph Visualization ---
st.header("🕸️ Entity Relationship Network Graph")

# Import necessary libraries for network graph
import networkx as nx
import plotly.graph_objects as go

def plot_entity_relationship_graph(df, min_connections=2):
    """Create a network graph visualization of entity relationships."""
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for users and IPs
    if 'user_name' in df.columns:
        for user in df['user_name'].unique():
            if pd.notna(user):
                G.add_node(str(user), type='user')
    
    if 'source_ip' in df.columns:
        for ip in df['source_ip'].unique():
            if pd.notna(ip):
                G.add_node(str(ip), type='ip')
    
    # Add edges between users and IPs
    if 'user_name' in df.columns and 'source_ip' in df.columns:
        # Count occurrences of each user-IP pair
        edge_counts = df.groupby(['user_name', 'source_ip']).size().reset_index(name='count')
        
        # Add edges for pairs that occur more than threshold times
        for _, row in edge_counts[edge_counts['count'] >= min_connections].iterrows():
            if pd.notna(row['user_name']) and pd.notna(row['source_ip']):
                G.add_edge(str(row['user_name']), str(row['source_ip']), weight=row['count'])
    
    # If graph is empty, return None
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return None
    
    # Create positions for nodes using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G.edges[edge]['weight']
        edge_text.append(f"Connections: {weight}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')
    
    # Create node traces for different types
    user_nodes_x = []
    user_nodes_y = []
    user_nodes_text = []
    
    ip_nodes_x = []
    ip_nodes_y = []
    ip_nodes_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        if G.nodes[node].get('type') == 'user':
            user_nodes_x.append(x)
            user_nodes_y.append(y)
            user_nodes_text.append(f"User: {node}")
        else:  # IP
            ip_nodes_x.append(x)
            ip_nodes_y.append(y)
            ip_nodes_text.append(f"IP: {node}")
    
    user_node_trace = go.Scatter(
        x=user_nodes_x, y=user_nodes_y,
        mode='markers',
        hoverinfo='text',
        text=user_nodes_text,
        marker=dict(
            color='blue',
            size=15,
            line_width=2))
    
    ip_node_trace = go.Scatter(
        x=ip_nodes_x, y=ip_nodes_y,
        mode='markers',
        hoverinfo='text',
        text=ip_nodes_text,
        marker=dict(
            color='red',
            size=15,
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, user_node_trace, ip_node_trace],
                  layout=go.Layout(
                      title='Entity Relationship Graph (Users and IPs)',
                      titlefont_size=16,
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig

# Add a slider for minimum connections threshold
min_connections = st.slider("Minimum Connections Threshold", 
                          min_value=1, max_value=10, 
                          value=2, step=1,
                          help="Minimum number of connections between entities to show in the graph")

# Generate and display the graph
if 'user_name' in df_processed.columns and 'source_ip' in df_processed.columns:
    graph_fig = plot_entity_relationship_graph(df_processed, min_connections)
    if graph_fig is not None:
        st.plotly_chart(graph_fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Graph Legend:**
        - **Blue nodes**: Users/Roles
        - **Red nodes**: IP Addresses
        - **Edges**: Connections between users and IPs
        - **Edge thickness**: Number of connections
        
        This graph helps identify unusual relationships between entities. Look for:
        - Users connecting from many different IPs
        - IPs used by many different users
        - Isolated clusters that might indicate separate attack campaigns
        """)
    else:
        st.info("Not enough entity relationship data to generate a graph with the current threshold.")
else:
    st.info("User or IP information not available in the data for relationship graph visualization.")

# --- Dashboard Info ---
st.sidebar.info(f"Dashboard displaying processed {data_type.capitalize()} logs and predictions from {selected_model} model.")

logging.info("Dashboard setup complete.")

# To run: streamlit run src/visualization/dashboard.py