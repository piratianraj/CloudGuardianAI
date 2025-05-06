# Threat Intelligence integration for VPC Flow Logs

import pandas as pd
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [THREAT_INTEL] - %(message)s')

def enrich_with_threat_intel(df, threat_intel_file=None):
    """
    Enrich VPC Flow Logs with threat intelligence data.
    
    Args:
        df: DataFrame containing VPC Flow Log data
        threat_intel_file: Optional path to a file containing malicious IPs (one per line)
        
    Returns:
        DataFrame with added threat intelligence flags
    """
    logging.info("Enriching VPC Flow Logs with threat intelligence data...")
    
    # If no threat intel file provided, use a simple example list
    if threat_intel_file is None:
        # Example malicious IPs (in a real implementation, this would come from a file or API)
        malicious_ips = [
            '123.45.67.89',  # Example malicious IP
            '98.76.54.32',  # Example C2 server
            '111.222.333.444'  # Example known scanner
        ]
    else:
        # Load threat intel from file
        try:
            with open(threat_intel_file, 'r') as f:
                malicious_ips = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(malicious_ips)} malicious IPs from threat intel file")
        except Exception as e:
            logging.error(f"Error loading threat intel file: {e}")
            malicious_ips = []
    
    # Flag connections to/from known malicious IPs
    if 'srcaddr' in df.columns:
        df['src_is_malicious'] = df['srcaddr'].isin(malicious_ips)
    else:
        df['src_is_malicious'] = False
        
    if 'dstaddr' in df.columns:
        df['dst_is_malicious'] = df['dstaddr'].isin(malicious_ips)
    else:
        df['dst_is_malicious'] = False
    
    # Overall threat flag
    df['has_threat_intel_match'] = df['src_is_malicious'] | df['dst_is_malicious']
    
    # Count how many connections were flagged
    flagged_count = df['has_threat_intel_match'].sum()
    logging.info(f"Flagged {flagged_count} connections with threat intelligence matches")
    
    return df

# Optional: Function to update threat intelligence from external sources
def update_threat_intel(output_file, api_key=None):
    """
    Updates threat intelligence from external sources.
    This is a placeholder function that would typically connect to threat intel APIs.
    
    Args:
        output_file: File to save the updated threat intel
        api_key: Optional API key for threat intel services
        
    Returns:
        Boolean indicating success or failure
    """
    logging.info("Updating threat intelligence data...")
    
    # In a real implementation, this would connect to threat intel APIs
    # For this example, we'll just create a sample file
    try:
        # Example malicious IPs
        malicious_ips = [
            '123.45.67.89',
            '98.76.54.32',
            '111.222.333.444',
            '10.20.30.40',
            '50.60.70.80'
        ]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write to file
        with open(output_file, 'w') as f:
            for ip in malicious_ips:
                f.write(f"{ip}\n")
                
        logging.info(f"Saved {len(malicious_ips)} malicious IPs to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error updating threat intel: {e}")
        return False