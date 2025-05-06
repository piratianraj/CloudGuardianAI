# Implementation of Graph-Based Analysis models for cloud security threat detection
# Includes: Graph Neural Networks, Graph Representation Learning, and Graph Algorithms

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# For Graph Neural Networks - these imports would need to be installed
# Uncomment when dependencies are installed
# import torch
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv
# from torch_geometric.data import Data, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [GRAPH_ML] - %(message)s')

# ===== GRAPH REPRESENTATION =====

def build_cloud_activity_graph(df, entity_cols=['userIdentity.arn', 'sourceIPAddress'], 
                              action_col='eventName', timestamp_col='eventTime'):
    """
    Builds a graph representation of cloud activity where nodes are entities (users, roles, IPs)
    and edges represent interactions or relationships.
    
    Args:
        df: DataFrame containing cloud activity logs (e.g., CloudTrail)
        entity_cols: Columns representing entities to be used as nodes
        action_col: Column representing the action performed
        timestamp_col: Column representing the timestamp of the event
        
    Returns:
        G: NetworkX graph object representing the cloud activity
    """
    logging.info("Building cloud activity graph...")
    
    try:
        # Initialize graph
        G = nx.DiGraph()
        
        # Process each row to build the graph
        for idx, row in df.iterrows():
            # Extract entities to create nodes
            entities = []
            for col in entity_cols:
                if pd.notna(row.get(col)):
                    entity_val = str(row[col])
                    entity_type = col.split('.')[-1]  # Extract the type (e.g., 'arn', 'sourceIPAddress')
                    entities.append((entity_val, {'type': entity_type}))
            
            # Add nodes if they don't exist
            for entity, attrs in entities:
                if not G.has_node(entity):
                    G.add_node(entity, **attrs)
            
            # Add edges between entities
            if len(entities) > 1:
                for i in range(len(entities) - 1):
                    source, _ = entities[i]
                    for j in range(i + 1, len(entities)):
                        target, _ = entities[j]
                        
                        # Create edge with attributes
                        edge_attrs = {
                            'action': row.get(action_col, 'unknown'),
                            'timestamp': row.get(timestamp_col, 'unknown'),
                            'weight': 1  # Initial weight
                        }
                        
                        # If edge already exists, update weight
                        if G.has_edge(source, target):
                            G[source][target]['weight'] += 1
                        else:
                            G.add_edge(source, target, **edge_attrs)
        
        logging.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    except Exception as e:
        logging.error(f"Error building cloud activity graph: {e}")
        return None

# ===== GRAPH ALGORITHMS =====

def detect_anomalies_with_graph_algorithms(G, centrality_threshold=0.9, community_outlier_threshold=0.8):
    """
    Detects anomalies in the cloud activity graph using graph algorithms.
    
    Args:
        G: NetworkX graph object representing the cloud activity
        centrality_threshold: Threshold for centrality measures (percentile)
        community_outlier_threshold: Threshold for community outlier detection
        
    Returns:
        anomalies: Dictionary of detected anomalies by different methods
    """
    logging.info("Detecting anomalies using graph algorithms...")
    
    try:
        anomalies = {}
        
        # 1. Centrality-based anomalies
        # Entities with unusually high centrality might indicate compromised accounts or pivotal points
        
        # Betweenness centrality - nodes that act as bridges between communities
        betweenness = nx.betweenness_centrality(G)
        threshold = np.percentile(list(betweenness.values()), centrality_threshold * 100)
        anomalies['high_betweenness'] = [node for node, score in betweenness.items() if score > threshold]
        
        # Degree centrality - nodes with unusually high connections
        in_degree = dict(G.in_degree(weight='weight'))
        threshold = np.percentile(list(in_degree.values()), centrality_threshold * 100)
        anomalies['high_in_degree'] = [node for node, score in in_degree.items() if score > threshold]
        
        # 2. Community detection and outlier identification
        # Detect communities and find nodes that bridge between multiple communities
        communities = nx.community.greedy_modularity_communities(G.to_undirected())
        
        # Create a mapping of node to community
        node_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_community[node] = i
        
        # Find nodes that have connections across communities
        community_bridges = []
        for node in G.nodes():
            if node in node_community:
                neighbors = list(G.neighbors(node))
                neighbor_communities = [node_community.get(neigh, -1) for neigh in neighbors if neigh in node_community]
                unique_communities = set(neighbor_communities)
                if len(unique_communities) > 1:  # Node connects to multiple communities
                    community_bridges.append((node, len(unique_communities)))
        
        # Sort by number of communities connected
        community_bridges.sort(key=lambda x: x[1], reverse=True)
        
        # Take top bridges as anomalies
        if community_bridges:
            threshold = np.percentile([score for _, score in community_bridges], community_outlier_threshold * 100)
            anomalies['community_bridges'] = [node for node, score in community_bridges if score > threshold]
        
        # 3. Temporal analysis (if timestamps are available)
        # Detect sudden changes in graph structure over time
        # This would require time-series analysis of graph metrics
        
        logging.info(f"Detected anomalies: {sum(len(v) for v in anomalies.values())} total across {len(anomalies)} methods")
        return anomalies
    
    except Exception as e:
        logging.error(f"Error detecting anomalies with graph algorithms: {e}")
        return None

# ===== GRAPH NEURAL NETWORKS =====

def prepare_graph_data_for_gnn(G, feature_df, node_feature_cols):
    """
    Prepares graph data for input to Graph Neural Networks.
    This is a placeholder function that would need to be implemented with actual GNN libraries.
    
    Args:
        G: NetworkX graph object
        feature_df: DataFrame containing node features
        node_feature_cols: Columns to use as node features
        
    Returns:
        graph_data: Data object for GNN input (format depends on the GNN library used)
    """
    logging.info("Preparing graph data for GNN...")
    logging.warning("This is a placeholder function. Actual implementation requires PyTorch Geometric or similar library.")
    
    # This would be implemented with PyTorch Geometric or similar library
    # Example pseudocode:
    # 1. Create node feature matrix
    # 2. Create edge index tensor
    # 3. Create edge attribute tensor
    # 4. Return Data object
    
    return None

def train_graph_neural_network(graph_data, epochs=100):
    """
    Trains a Graph Neural Network for anomaly detection.
    This is a placeholder function that would need to be implemented with actual GNN libraries.
    
    Args:
        graph_data: Prepared graph data for GNN
        epochs: Number of training epochs
        
    Returns:
        model: Trained GNN model
    """
    logging.info("Training Graph Neural Network...")
    logging.warning("This is a placeholder function. Actual implementation requires PyTorch Geometric or similar library.")
    
    # This would be implemented with PyTorch Geometric or similar library
    # Example pseudocode:
    # 1. Define GNN model architecture
    # 2. Set up optimizer
    # 3. Train model
    # 4. Return trained model
    
    return None

# ===== TEMPORAL GRAPH ANALYSIS =====

def analyze_temporal_graph_evolution(df, entity_cols, action_col, timestamp_col, time_window='1H'):
    """
    Analyzes how the graph structure evolves over time to detect gradual privilege accumulation.
    
    Args:
        df: DataFrame containing cloud activity logs
        entity_cols: Columns representing entities to be used as nodes
        action_col: Column representing the action performed
        timestamp_col: Column representing the timestamp of the event
        time_window: Time window for creating temporal graphs
        
    Returns:
        temporal_anomalies: Dictionary of detected temporal anomalies
    """
    logging.info(f"Analyzing temporal graph evolution with window {time_window}...")
    
    try:
        # Ensure timestamp column is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Group data by time windows
        df = df.sort_values(by=timestamp_col)
        time_groups = df.groupby(pd.Grouper(key=timestamp_col, freq=time_window))
        
        # Track graph metrics over time
        temporal_metrics = []
        
        # Build graph for each time window
        for time, group in time_groups:
            if len(group) > 0:
                # Build graph for this time window
                G = build_cloud_activity_graph(group, entity_cols, action_col, timestamp_col)
                
                if G and G.number_of_nodes() > 0:
                    # Calculate graph metrics
                    metrics = {
                        'timestamp': time,
                        'num_nodes': G.number_of_nodes(),
                        'num_edges': G.number_of_edges(),
                        'density': nx.density(G),
                        'avg_clustering': nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 2 else 0
                    }
                    
                    # Add centrality metrics for key nodes if graph is not too large
                    if G.number_of_nodes() < 1000:  # Limit for performance reasons
                        try:
                            betweenness = nx.betweenness_centrality(G)
                            if betweenness:
                                metrics['max_betweenness'] = max(betweenness.values())
                        except:
                            pass
                    
                    temporal_metrics.append(metrics)
        
        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame(temporal_metrics)
        
        if len(metrics_df) > 1:
            # Detect anomalies in temporal patterns
            temporal_anomalies = {}
            
            # Calculate rolling statistics
            for col in ['num_nodes', 'num_edges', 'density', 'max_betweenness']:
                if col in metrics_df.columns:
                    # Calculate rolling mean and std
                    window_size = min(5, len(metrics_df) - 1)  # Use smaller window if not enough data
                    if window_size > 0:
                        metrics_df[f'{col}_rolling_mean'] = metrics_df[col].rolling(window=window_size, min_periods=1).mean()
                        metrics_df[f'{col}_rolling_std'] = metrics_df[col].rolling(window=window_size, min_periods=1).std()
                        
                        # Detect significant changes (> 2 std from rolling mean)
                        metrics_df[f'{col}_zscore'] = (metrics_df[col] - metrics_df[f'{col}_rolling_mean']) / metrics_df[f'{col}_rolling_std'].replace(0, 1)
                        
                        # Identify anomalies
                        anomalies = metrics_df[abs(metrics_df[f'{col}_zscore']) > 2]
                        if not anomalies.empty:
                            temporal_anomalies[col] = anomalies[['timestamp', col, f'{col}_zscore']].to_dict('records')
            
            logging.info(f"Detected {sum(len(v) for v in temporal_anomalies.values())} temporal anomalies")
            return temporal_anomalies
        else:
            logging.warning("Not enough temporal data points for analysis")
            return {}
    
    except Exception as e:
        logging.error(f"Error analyzing temporal graph evolution: {e}")
        return {}

# ===== KNOWLEDGE GRAPH INTEGRATION =====

def build_security_knowledge_graph():
    """
    Builds a security knowledge graph with known attack patterns and relationships.
    This is a placeholder function that would create a basic security knowledge graph.
    
    Returns:
        KG: NetworkX graph object representing the security knowledge graph
    """
    logging.info("Building security knowledge graph...")
    
    try:
        # Initialize knowledge graph
        KG = nx.DiGraph()
        
        # Add known attack patterns
        # This would be populated with actual security knowledge
        # Example: Privilege Escalation Pattern
        KG.add_node('ListUsers', type='API', category='Discovery')
        KG.add_node('GetUserPolicy', type='API', category='Discovery')
        KG.add_node('PutUserPolicy', type='API', category='PrivilegeEscalation')
        KG.add_node('CreateAccessKey', type='API', category='CredentialAccess')
        
        KG.add_edge('ListUsers', 'GetUserPolicy', relationship='followed_by', risk=0.3)
        KG.add_edge('GetUserPolicy', 'PutUserPolicy', relationship='followed_by', risk=0.7)
        KG.add_edge('PutUserPolicy', 'CreateAccessKey', relationship='followed_by', risk=0.8)
        
        # Add more patterns for different attack techniques
        # ...
        
        logging.info(f"Knowledge graph built with {KG.number_of_nodes()} nodes and {KG.number_of_edges()} edges")
        return KG
    
    except Exception as e:
        logging.error(f"Error building security knowledge graph: {e}")
        return None

def match_activity_with_knowledge_graph(activity_graph, knowledge_graph):
    """
    Matches observed activity patterns with known patterns in the security knowledge graph.
    
    Args:
        activity_graph: NetworkX graph of observed cloud activity
        knowledge_graph: Security knowledge graph with known attack patterns
        
    Returns:
        matches: Dictionary of matched patterns and their risk scores
    """
    logging.info("Matching activity with security knowledge graph...")
    
    try:
        matches = {}
        
        # This is a simplified implementation
        # A real implementation would use subgraph isomorphism or similar techniques
        
        # Check for API sequences that match known patterns
        for node in activity_graph.nodes():
            # Skip non-API nodes
            if activity_graph.nodes[node].get('type') != 'API':
                continue
                
            # Look for matches in knowledge graph
            if knowledge_graph.has_node(node):
                # Check for sequence matches
                for successor in activity_graph.successors(node):
                    if knowledge_graph.has_edge(node, successor):
                        pattern = f"{node}->{successor}"
                        risk = knowledge_graph[node][successor].get('risk', 0.5)
                        matches[pattern] = risk
        
        logging.info(f"Found {len(matches)} matches with knowledge graph patterns")
        return matches
    
    except Exception as e:
        logging.error(f"Error matching with knowledge graph: {e}")
        return {}

# ===== VISUALIZATION =====

def visualize_cloud_activity_graph(G, output_path=None, highlight_nodes=None):
    """
    Visualizes the cloud activity graph with optional highlighting of anomalous nodes.
    
    Args:
        G: NetworkX graph object
        output_path: Path to save the visualization (if None, display only)
        highlight_nodes: List of nodes to highlight as anomalous
        
    Returns:
        fig: Matplotlib figure object
    """
    logging.info("Visualizing cloud activity graph...")
    
    try:
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(G, seed=42)  # For reproducibility
        
        # Draw regular nodes and edges
        node_colors = ['skyblue' if node not in (highlight_nodes or []) else 'red' for node in G.nodes()]
        node_sizes = [100 if node not in (highlight_nodes or []) else 300 for node in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, arrows=True)
        
        # Add labels to important nodes
        if highlight_nodes:
            labels = {node: node for node in G.nodes() if node in highlight_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title('Cloud Activity Graph Analysis')
        plt.axis('off')
        
        # Save or display
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logging.info(f"Graph visualization saved to {output_path}")
        
        return plt.gcf()
    
    except Exception as e:
        logging.error(f"Error visualizing graph: {e}")
        return None

# ===== MAIN FUNCTIONS =====

def perform_graph_based_analysis(df, entity_cols=['userIdentity.arn', 'sourceIPAddress'], 
                               action_col='eventName', timestamp_col='eventTime',
                               output_dir=None):
    """
    Performs comprehensive graph-based analysis on cloud activity data.
    
    Args:
        df: DataFrame containing cloud activity logs
        entity_cols: Columns representing entities to be used as nodes
        action_col: Column representing the action performed
        timestamp_col: Column representing the timestamp of the event
        output_dir: Directory to save outputs (visualizations, results)
        
    Returns:
        results: Dictionary containing analysis results
    """
    logging.info("Performing comprehensive graph-based analysis...")
    
    results = {}
    
    try:
        # 1. Build activity graph
        G = build_cloud_activity_graph(df, entity_cols, action_col, timestamp_col)
        if G is None:
            return {"error": "Failed to build activity graph"}
        
        results["graph_stats"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_weakly_connected(G)
        }
        
        # 2. Detect anomalies using graph algorithms
        algorithm_anomalies = detect_anomalies_with_graph_algorithms(G)
        if algorithm_anomalies:
            results["algorithm_anomalies"] = algorithm_anomalies
        
        # 3. Perform temporal analysis
        temporal_anomalies = analyze_temporal_graph_evolution(df, entity_cols, action_col, timestamp_col)
        if temporal_anomalies:
            results["temporal_anomalies"] = temporal_anomalies
        
        # 4. Knowledge graph integration
        KG = build_security_knowledge_graph()
        if KG:
            kg_matches = match_activity_with_knowledge_graph(G, KG)
            if kg_matches:
                results["knowledge_graph_matches"] = kg_matches
        
        # 5. Visualize results if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all anomalous nodes
            anomalous_nodes = set()
            if "algorithm_anomalies" in results:
                for method, nodes in results["algorithm_anomalies"].items():
                    anomalous_nodes.update(nodes)
            
            # Visualize graph with anomalies highlighted
            fig = visualize_cloud_activity_graph(
                G, 
                output_path=os.path.join(output_dir, "cloud_activity_graph.png"),
                highlight_nodes=list(anomalous_nodes)
            )
        
        logging.info("Graph-based analysis complete")
        return results
    
    except Exception as e:
        logging.error(f"Error in graph-based analysis: {e}")
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    logging.info("This module provides graph-based analysis capabilities for cloud security")
    logging.info("Import and use the functions in your main application")