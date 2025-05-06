# Processes raw CloudTrail and VPC Flow logs and performs initial feature engineering

import pandas as pd
import numpy as np
import json
import os
import gzip
import logging
import argparse
from glob import glob
import csv # Added for VPC logs

# Import threat intelligence module
from .threat_intel import enrich_with_threat_intel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [PROCESSING] - %(message)s')

# --- CloudTrail Parsing ---

def parse_cloudtrail_file(file_path):
    """Parses a single CloudTrail log file (.json or .json.gz)."""
    records = []
    try:
        logging.debug(f"Processing CloudTrail file: {file_path}")
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            log_data = json.load(f)
            records = log_data.get('Records', [])
            if not isinstance(records, list):
                logging.warning(f"'Records' key in {file_path} is not a list. Skipping.")
                return []
        logging.debug(f"Successfully parsed {len(records)} CloudTrail records from {file_path}")
        return records
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error in CloudTrail file {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error processing CloudTrail file {file_path}: {e}")
        return []

# --- VPC Flow Log Parsing ---

def parse_vpc_flow_log_file(file_path):
    """Parses a single VPC Flow Log file (.log or .log.gz)."""
    try:
        logging.debug(f"Processing VPC Flow Log file: {file_path}")
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            # Read header to get column names
            header = f.readline().strip()
            if not header:
                logging.warning(f"Empty or missing header in VPC Flow Log file: {file_path}")
                return None
            column_names = header.split(' ')

            # Use pandas read_csv for efficient parsing
            # Need to skip the header row we already read
            df = pd.read_csv(f, sep=' ', header=None, names=column_names,
                             on_bad_lines='warn', # Log problematic lines
                             quoting=csv.QUOTE_NONE) # Avoid issues with potential quotes

        logging.debug(f"Successfully parsed {len(df)} VPC Flow Log records from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"VPC Flow Log file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing VPC Flow Log file {file_path}: {e}")
        return None

# --- Main Processing Logic ---

def process_logs(raw_log_dir, processed_data_dir):
    """Reads raw logs (CloudTrail & VPC), parses them, and saves flattened DataFrames."""
    logging.info(f"Starting log processing from directory: {raw_log_dir}")

    # Ensure output directory exists
    try:
        os.makedirs(processed_data_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {processed_data_dir}: {e}")
        return

    all_cloudtrail_records = []
    all_vpc_dfs = []

    # Find all relevant log files recursively
    # Adjust glob pattern if needed based on actual filenames
    log_files = glob(os.path.join(raw_log_dir, '**', '*.json*'), recursive=True) + \
                glob(os.path.join(raw_log_dir, '**', '*.log*'), recursive=True)

    # Remove duplicates if patterns overlap (e.g., .log.gz caught by both)
    log_files = sorted(list(set(log_files)))

    if not log_files:
        logging.warning(f"No log files (.json*, .log*) found in {raw_log_dir}")
        return

    logging.info(f"Found {len(log_files)} potential log files to process.")

    for file_path in log_files:
        filename = os.path.basename(file_path)
        # Basic check to differentiate log types based on common naming conventions
        if 'cloudtrail' in filename.lower() and (file_path.endswith('.json') or file_path.endswith('.json.gz')):
            records = parse_cloudtrail_file(file_path)
            all_cloudtrail_records.extend(records)
        elif 'vpc' in filename.lower() and (file_path.endswith('.log') or file_path.endswith('.log.gz')):
            df_vpc = parse_vpc_flow_log_file(file_path)
            if df_vpc is not None:
                all_vpc_dfs.append(df_vpc)
        else:
            logging.debug(f"Skipping file (unknown type or extension): {file_path}")

    # --- Process CloudTrail Logs ---
    if not all_cloudtrail_records:
        logging.warning("No valid CloudTrail records found.")
    else:
        logging.info(f"Total CloudTrail records parsed: {len(all_cloudtrail_records)}")
        try:
            df_ct = pd.json_normalize(all_cloudtrail_records)
            logging.info(f"Created CloudTrail DataFrame with shape: {df_ct.shape}")
            df_ct = process_cloudtrail_features(df_ct) # Apply CloudTrail specific features
            save_dataframe(df_ct, processed_data_dir, 'processed_cloudtrail_logs.parquet')
        except Exception as e:
            logging.error(f"Failed to process or save CloudTrail DataFrame: {e}")

    # --- Process VPC Flow Logs ---
    if not all_vpc_dfs:
        logging.warning("No valid VPC Flow Log records found.")
    else:
        logging.info(f"Total VPC Flow Log DataFrames parsed: {len(all_vpc_dfs)}")
        try:
            df_vpc_all = pd.concat(all_vpc_dfs, ignore_index=True)
            logging.info(f"Combined VPC Flow Log DataFrame with shape: {df_vpc_all.shape}")
            df_vpc_all = process_vpc_features(df_vpc_all) # Apply VPC specific features
            save_dataframe(df_vpc_all, processed_data_dir, 'processed_vpc_flow_logs.parquet')
        except Exception as e:
            logging.error(f"Failed to process or save VPC Flow Log DataFrame: {e}")

# --- Feature Engineering Functions ---

def process_cloudtrail_features(df):
    """Applies feature engineering specific to CloudTrail data."""
    logging.info("Performing CloudTrail cleaning and feature engineering...")

    # Convert eventTime to datetime objects
    if 'eventTime' in df.columns:
        df['eventTime'] = pd.to_datetime(df['eventTime'], errors='coerce')
        original_count = len(df)
        df.dropna(subset=['eventTime'], inplace=True)
        if len(df) < original_count:
            logging.warning(f"Dropped {original_count - len(df)} CloudTrail rows due to invalid eventTime.")

    # Extract simple features
    if 'userIdentity.type' in df.columns: df['user_type'] = df['userIdentity.type']
    if 'userIdentity.userName' in df.columns: df['user_name'] = df['userIdentity.userName']
    if 'user_name' not in df.columns and 'userIdentity.arn' in df.columns: df['user_name'] = df['userIdentity.arn']
    if 'sourceIPAddress' in df.columns: df['source_ip'] = df['sourceIPAddress']
    if 'eventTime' in df.columns:
        df['event_hour'] = df['eventTime'].dt.hour
        df['event_day_of_week'] = df['eventTime'].dt.dayofweek
    if 'errorCode' in df.columns: df['is_error'] = df['errorCode'].notna()
    else: df['is_error'] = False
    if 'userAgent' in df.columns: df['user_agent_present'] = df['userAgent'].notna()
    else: df['user_agent_present'] = False
    # Add awsRegion if available
    if 'awsRegion' in df.columns: df['awsRegion'] = df['awsRegion']
    # Add eventName if available
    if 'eventName' in df.columns: df['eventName'] = df['eventName']

    # Frequency Analysis
    logging.info("Calculating CloudTrail frequency features...")
    if 'user_name' in df.columns and 'eventName' in df.columns:
        df['events_per_user'] = df.groupby('user_name')['eventName'].transform('count')
    else: df['events_per_user'] = 0
    if 'source_ip' in df.columns and 'eventName' in df.columns:
        df['events_per_ip'] = df.groupby('source_ip')['eventName'].transform('count')
    else: df['events_per_ip'] = 0
    if 'user_name' in df.columns and 'eventName' in df.columns:
        df['console_logins_per_user'] = df[df['eventName'] == 'ConsoleLogin'].groupby('user_name')['eventName'].transform('count')
        df['console_logins_per_user'] = df['console_logins_per_user'].fillna(0)
    else: df['console_logins_per_user'] = 0

    # Add is_simulated_attack if present from generation
    if 'is_simulated_attack' in df.columns:
        df['is_simulated_attack'] = df['is_simulated_attack'].fillna(False).astype(bool)
    else:
        df['is_simulated_attack'] = False # Assume normal if label not present

    logging.info("CloudTrail feature engineering complete.")
    return df

def process_vpc_features(df):
    """Applies feature engineering specific to VPC Flow Log data."""
    logging.info("Performing VPC Flow Log cleaning and feature engineering...")

    # Convert timestamp columns to numeric/datetime
    for col in ['start', 'end']: # Assuming 'start', 'end' are standard VPC Flow Log v2+ field names
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Optionally convert to datetime, though epoch seconds might be fine for ML
            # df[f'{col}_dt'] = pd.to_datetime(df[col], unit='s', errors='coerce')

    # Convert numeric columns
    for col in ['version', 'protocol', 'packets', 'bytes', 'srcport', 'dstport']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaNs created by coercion (e.g., '-' in ports for ICMP)
            if col in ['srcport', 'dstport']:
                df[col] = df[col].fillna(-1) # Use -1 for missing/non-applicable ports
            else:
                df[col] = df[col].fillna(0)

    # Calculate flow duration
    if 'start' in df.columns and 'end' in df.columns:
        df['flow_duration'] = df['end'] - df['start']
        df['flow_duration'] = df['flow_duration'].fillna(0).clip(lower=0) # Ensure non-negative duration

    # Map protocol numbers to names (optional, could be useful)
    protocol_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
    if 'protocol' in df.columns:
        df['protocol_name'] = df['protocol'].map(protocol_map).fillna('Other')

    # Add placeholder for is_simulated_attack (if generated with labels)
    # This requires the simulation script to add a label column/indicator
    if 'is_attack' in df.columns: # Assuming simulation adds 'is_attack'
         df['is_simulated_attack'] = df['is_attack'].astype(bool)
    # Read the label generated by the simulation script
    if 'is_simulated_attack' in df.columns:
         # Ensure it's treated as integer/boolean after reading from CSV
         df['is_simulated_attack'] = pd.to_numeric(df['is_simulated_attack'], errors='coerce').fillna(0).astype(int)
    else:
         df['is_simulated_attack'] = False

    # Count unique ports/IPs contacted
    if 'srcaddr' in df.columns and 'dstaddr' in df.columns:
        # Group by source IP and count unique destination IPs
        src_ip_counts = df.groupby('srcaddr')['dstaddr'].nunique().to_dict()
        df['unique_dest_ips'] = df['srcaddr'].map(src_ip_counts)
        
        # Group by source IP and count unique destination ports
        if 'dstport' in df.columns:
            src_port_counts = df.groupby('srcaddr')['dstport'].nunique().to_dict()
            df['unique_dest_ports'] = df['srcaddr'].map(src_port_counts)

    # Calculate inbound/outbound traffic ratio
    if 'srcaddr' in df.columns and 'dstaddr' in df.columns and 'bytes' in df.columns:
        # For each IP, calculate total bytes sent and received
        ip_outbound = df.groupby('srcaddr')['bytes'].sum().to_dict()
        ip_inbound = df.groupby('dstaddr')['bytes'].sum().to_dict()
        
        # Add ratio to rows where this IP appears as source
        for ip in set(df['srcaddr'].unique()):
            outbound = ip_outbound.get(ip, 0)
            inbound = ip_inbound.get(ip, 0)
            
            # Avoid division by zero
            ratio = outbound / inbound if inbound > 0 else float('inf')
            
            # Add ratio to rows where this IP appears
            df.loc[df['srcaddr'] == ip, 'outbound_inbound_ratio'] = ratio

    # Periodicity/beaconing detection
    if 'start' in df.columns:
        # Group by source-destination pair
        for (src, dst), group in df.groupby(['srcaddr', 'dstaddr']):
            if len(group) >= 3:  # Need at least 3 points to detect periodicity
                # Sort by timestamp
                sorted_times = sorted(group['start'].values)
                
                # Calculate time differences between consecutive connections
                time_diffs = np.diff(sorted_times)
                
                # Calculate coefficient of variation (std/mean) - lower values indicate more regular patterns
                if len(time_diffs) > 0 and np.mean(time_diffs) > 0:
                    regularity = np.std(time_diffs) / np.mean(time_diffs)
                    
                    # Mark connections with high regularity (low CV) as potential beaconing
                    # CV < 0.1 is extremely regular, < 0.3 is suspicious
                    df.loc[(df['srcaddr'] == src) & (df['dstaddr'] == dst), 'beaconing_score'] = 1 - min(regularity, 1.0)

    # Add threat intelligence integration
    df = enrich_with_threat_intel(df)

    logging.info("VPC Flow Log feature engineering complete.")
    return df

# --- Utility Functions ---

def save_dataframe(df, output_dir, filename):
    """Saves a DataFrame to a Parquet file."""
    if df is None or df.empty:
        logging.warning(f"DataFrame for {filename} is empty or None. Skipping save.")
        return

    output_file = os.path.join(output_dir, filename)
    try:
        df.to_parquet(output_file, index=False)
        logging.info(f"Processed data saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to save processed data to {output_file}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw CloudTrail and VPC Flow logs.")
    parser.add_argument('--raw-dir', default='../../data/raw_logs', help="Directory containing raw log files (.json* for CloudTrail, .log* for VPC).")
    parser.add_argument('--processed-dir', default='../../data/processed', help="Directory to save the processed data (Parquet files).")

    args = parser.parse_args()

    process_logs(raw_log_dir=args.raw_dir, processed_data_dir=args.processed_dir)