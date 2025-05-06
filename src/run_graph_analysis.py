#!/usr/bin/env python3
# Script to run graph-based analysis on CloudTrail logs

import os
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
from src.ml_engine.graph_models import perform_graph_based_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create output directory for results
    output_dir = os.path.join('data', 'graph_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed CloudTrail logs
    try:
        logging.info("Loading processed CloudTrail logs...")
        data_path = os.path.join('data', 'processed', 'processed_cloudtrail_logs.parquet')
        df = pd.read_parquet(data_path)
        logging.info(f"Loaded {len(df)} log entries")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Run graph-based analysis
    logging.info("Running graph-based analysis...")
    results = perform_graph_based_analysis(
        df=df,
        entity_cols=['userIdentity.arn', 'sourceIPAddress'],
        action_col='eventName',
        timestamp_col='eventTime',
        output_dir=output_dir
    )
    
    # Save results to JSON file
    results_path = os.path.join(output_dir, 'analysis_results.json')
    
    # Convert non-serializable objects to strings
    serializable_results = {}
    for key, value in results.items():
        if key == 'graph_stats':
            serializable_results[key] = value
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, list):
                    serializable_results[key][k] = [str(item) for item in v]
                else:
                    serializable_results[key][k] = str(v)
        else:
            serializable_results[key] = str(value)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Analysis results saved to {results_path}")
    logging.info(f"Graph visualization saved to {os.path.join(output_dir, 'cloud_activity_graph.png')}")
    
    # Print summary of findings
    print("\nGraph Analysis Summary:")
    print(f"Graph Statistics: {len(df)} events analyzed")
    print(f"  - Nodes: {results.get('graph_stats', {}).get('num_nodes', 0)}")
    print(f"  - Edges: {results.get('graph_stats', {}).get('num_edges', 0)}")
    print(f"  - Density: {results.get('graph_stats', {}).get('density', 0):.6f}")
    
    if 'algorithm_anomalies' in results:
        total_anomalies = sum(len(nodes) for nodes in results['algorithm_anomalies'].values())
        print(f"Detected {total_anomalies} anomalies using graph algorithms")
        
        for method, nodes in results['algorithm_anomalies'].items():
            print(f"  - {method}: {len(nodes)} anomalous entities")
    
    print(f"\nResults and visualizations saved to: {output_dir}")
    print("To view the graph visualization, open the PNG file in your image viewer.")

if __name__ == "__main__":
    main()