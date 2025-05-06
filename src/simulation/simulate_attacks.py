# Simulates attack scenarios using Boto3

import boto3
import time
import random
import logging
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [ATTACK_SIM] - %(message)s')

# Initialize Boto3 clients (assumes credentials are configured)
DEFAULT_REGION = 'us-east-1' # Match the region used in normal simulation if possible
try:
    session = boto3.Session()
    ec2 = session.client('ec2', region_name=DEFAULT_REGION)
    iam = session.client('iam') # IAM is global, region is less critical but specify for consistency
    # Add other clients as needed
    logging.info(f"Boto3 clients initialized for attack simulation in region: {session.region_name or DEFAULT_REGION}")
except Exception as e:
    logging.error(f"Failed to initialize Boto3 clients: {e}")
    exit(1)

def simulate_discovery_recon(intensity=5):
    """Simulates reconnaissance by making excessive List/Describe calls."""
    logging.info(f"Simulating Discovery/Reconnaissance (Intensity: {intensity})...")
    actions = [
        lambda: ec2.describe_instances(),
        lambda: ec2.describe_security_groups(),
        lambda: ec2.describe_vpcs(),
        lambda: ec2.describe_subnets(),
        lambda: iam.list_users(),
        lambda: iam.list_roles(),
        # Add more list/describe calls for other services (S3, RDS, etc.)
    ]
    try:
        for _ in range(intensity):
            action = random.choice(actions)
            try:
                action()
                logging.debug(f"Executed recon action: {action.__name__}")
            except Exception as e:
                logging.warning(f"Recon action failed (might be expected): {e}")
            time.sleep(random.uniform(0.1, 0.5)) # Short delay between rapid calls
        logging.info("Discovery/Reconnaissance simulation step finished.")
    except Exception as e:
        logging.error(f"Error during Discovery/Recon simulation: {e}")

def simulate_persistence_iam_user():
    """Simulates creating a suspicious IAM user for persistence."""
    # Generate a suspicious-looking username
    username = 'bkdr_svc_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    logging.info(f"Simulating Persistence: Attempting to create IAM user '{username}'...")
    try:
        # In a real scenario, this would create the user.
        # For safety in simulation, we might just log the intent or use a specific naming convention.
        # iam.create_user(UserName=username)
        # logging.info(f"Successfully simulated creation of IAM user '{username}'.")
        # To avoid actually creating users during tests, we just log the attempt:
        logging.info(f"Simulated call to create IAM user: {username}")
        # Optionally, simulate adding the user to a group or attaching policies
        # iam.add_user_to_group(GroupName='AdminGroup', UserName=username) # Example
        logging.info(f"Simulated call to add user {username} to potentially privileged group.")

    except Exception as e:
        # This might fail due to permissions, which is also a valid log event
        logging.error(f"Error during IAM Persistence simulation for user '{username}': {e}")

# Add more attack simulation functions based on MITRE ATT&CK for Cloud TTPs
# Examples:
# - simulate_defense_evasion_stop_cloudtrail()
# - simulate_credential_access_assume_role_abuse()
# - simulate_exfiltration_s3_download()

def run_attack_simulation(duration_minutes=1, actions_per_minute=3):
    """Runs the attack simulation for a specified duration."""
    logging.info(f"Starting attack simulation for {duration_minutes} minutes.")
    end_time = time.time() + duration_minutes * 60
    possible_attacks = [
        lambda: simulate_discovery_recon(intensity=random.randint(3, 8)),
        simulate_persistence_iam_user,
        # Add other attack functions here
    ]

    while time.time() < end_time:
        attack_action = random.choice(possible_attacks)
        try:
            attack_action()
        except Exception as e:
            logging.error(f"Error executing attack action {attack_action.__name__}: {e}")

        # Wait interval
        sleep_time = 60 / actions_per_minute * (0.7 + random.random() * 0.6) # Jitter
        logging.debug(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    logging.info("Attack simulation finished.")

if __name__ == "__main__":
    # Example: Run simulation for 1 minute with approx 3 attack actions per minute
    run_attack_simulation(duration_minutes=1, actions_per_minute=3)