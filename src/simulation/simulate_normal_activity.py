# Simulates normal cloud activity using Boto3

import boto3
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Boto3 clients (assumes credentials are configured in the environment)
# Use a default region or allow Boto3 to determine it
DEFAULT_REGION = 'us-east-1' # Or choose a region relevant to your setup
try:
    session = boto3.Session()
    ec2 = session.client('ec2', region_name=DEFAULT_REGION)
    s3 = session.client('s3', region_name=DEFAULT_REGION)
    # Add other clients as needed (e.g., iam, rds)
    logging.info(f"Boto3 clients initialized in region: {session.region_name or DEFAULT_REGION}")
except Exception as e:
    logging.error(f"Failed to initialize Boto3 clients: {e}")
    exit(1)

def simulate_s3_activity():
    """Simulates common S3 read operations."""
    try:
        logging.info("Simulating S3: Listing buckets...")
        buckets = s3.list_buckets()
        logging.info(f"Found {len(buckets.get('Buckets', []))} buckets.")
        # Optionally, list objects in a random bucket if buckets exist
        if buckets.get('Buckets'):
            random_bucket = random.choice(buckets['Buckets'])['Name']
            try:
                logging.info(f"Simulating S3: Listing objects in bucket '{random_bucket}'...")
                s3.list_objects_v2(Bucket=random_bucket, MaxKeys=10)
            except Exception as e:
                # Catch exceptions for specific buckets (e.g., access denied, different region)
                logging.warning(f"Could not list objects in bucket '{random_bucket}': {e}")
    except Exception as e:
        logging.error(f"Error during S3 simulation: {e}")

def simulate_ec2_activity():
    """Simulates common EC2 read operations."""
    try:
        logging.info("Simulating EC2: Describing instances...")
        ec2.describe_instances(MaxResults=5)
        logging.info("Simulating EC2: Describing security groups...")
        ec2.describe_security_groups(MaxResults=5)
        logging.info("Simulating EC2: Describing VPCs...")
        ec2.describe_vpcs(MaxResults=5)
    except Exception as e:
        logging.error(f"Error during EC2 simulation: {e}")

# Add more simulation functions for other services (IAM, RDS, Lambda, etc.)
# def simulate_iam_activity():
#     try:
#         logging.info("Simulating IAM: Listing users...")
#         iam = session.client('iam')
#         iam.list_users(MaxItems=10)
#     except Exception as e:
#         logging.error(f"Error during IAM simulation: {e}")

def run_simulation(duration_minutes=1, actions_per_minute=5):
    """Runs the simulation for a specified duration."""
    logging.info(f"Starting normal activity simulation for {duration_minutes} minutes.")
    end_time = time.time() + duration_minutes * 60
    possible_actions = [
        simulate_s3_activity,
        simulate_ec2_activity,
        # simulate_iam_activity, # Add more actions here
    ]

    while time.time() < end_time:
        # Choose a random action
        action = random.choice(possible_actions)
        try:
            action()
        except Exception as e:
            logging.error(f"Error executing action {action.__name__}: {e}")

        # Wait for a random interval before the next action
        sleep_time = 60 / actions_per_minute * (0.5 + random.random()) # Add jitter
        logging.debug(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    logging.info("Normal activity simulation finished.")

if __name__ == "__main__":
    # Example: Run simulation for 2 minutes with approx 10 actions per minute
    run_simulation(duration_minutes=2, actions_per_minute=10)