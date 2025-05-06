# Generates sample CloudTrail-like log files locally

import json
import gzip
import os
import random
import uuid
import datetime
import ipaddress
import argparse
import logging
import time
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [LOG_GEN] - %(message)s')

# --- Sample Data --- 
SAMPLE_USERS = [
    {'type': 'IAMUser', 'principalId': 'AIDACKCEVSQ6C2EXAMPLE', 'arn': 'arn:aws:iam::123456789012:user/Alice', 'accountId': '123456789012', 'userName': 'Alice'},
    {'type': 'IAMUser', 'principalId': 'AIDAPI4ABCDEFGHIJKL', 'arn': 'arn:aws:iam::123456789012:user/Bob', 'accountId': '123456789012', 'userName': 'Bob'},
    {'type': 'AssumedRole', 'principalId': 'AROACKCEVSQ6C2EXAMPLE:role-session', 'arn': 'arn:aws:sts::123456789012:assumed-role/AdminRole/role-session', 'accountId': '123456789012', 'sessionContext': {'attributes': {'mfaAuthenticated': 'false', 'creationDate': '...'}, 'sessionIssuer': {'type': 'Role', 'principalId': 'AROACKCEVSQ6C2EXAMPLE', 'arn': 'arn:aws:iam::123456789012:role/AdminRole', 'accountId': '123456789012', 'userName': 'AdminRole'}}},
    {'type': 'Root', 'principalId': '123456789012', 'arn': 'arn:aws:iam::123456789012:root', 'accountId': '123456789012'}
]
SAMPLE_REGIONS = ['us-east-1', 'us-west-2', 'eu-central-1', 'ap-southeast-2']
SAMPLE_EVENT_SOURCES = ['ec2.amazonaws.com', 'iam.amazonaws.com', 's3.amazonaws.com', 'signin.amazonaws.com']
SAMPLE_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'aws-sdk-java/1.11.106 Linux/4.9.124-0.1.ac.198.71.329.metal1.x86_64 Java_HotSpot(TM)_64-Bit_Server_VM/25.152-b16 Java/1.8.0_152',
    'Boto3/1.26.42 Python/3.9.11 Linux/5.4.0-109-generic Botocore/1.29.42',
    'AWS CLI/2.9.10 Python/3.9.11 Windows/10 exe/AMD64 prompt/off'
]

# --- Log Entry Generation Functions --- 

def generate_base_event(event_name, event_source, region, user_identity, source_ip, is_attack=False):
    """Generates the common structure for a CloudTrail event."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        'eventVersion': '1.08',
        'userIdentity': user_identity,
        'eventTime': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'eventSource': event_source,
        'eventName': event_name,
        'awsRegion': region,
        'sourceIPAddress': source_ip,
        'userAgent': random.choice(SAMPLE_USER_AGENTS),
        'requestParameters': None, # Specific to event
        'responseElements': None, # Specific to event
        'eventID': str(uuid.uuid4()),
        'readOnly': event_name.startswith(('List', 'Describe', 'Get')),
        'eventType': 'AwsApiCall', # Or 'AwsConsoleSignIn'
        'recipientAccountId': user_identity['accountId'],
        'is_simulated_attack': is_attack # Add the label
    }

def generate_console_login(success=True, is_attack=False):
    """Generates a sample ConsoleLogin event."""
    user = random.choice(SAMPLE_USERS)
    region = random.choice(SAMPLE_REGIONS)
    source_ip = str(ipaddress.IPv4Address(random.randint(1, 0xFFFFFFFF)))
    # Mark failed logins as attack-like if is_attack is True
    event = generate_base_event('ConsoleLogin', 'signin.amazonaws.com', region, user, source_ip, is_attack=(is_attack and not success))
    event['eventType'] = 'AwsConsoleSignIn'
    event['responseElements'] = {'ConsoleLogin': 'Success' if success else 'Failure'}
    if not success:
        event['errorMessage'] = 'Failed authentication'
    return event

def generate_describe_instances(is_attack=False):
    """Generates a sample DescribeInstances event."""
    user = random.choice(SAMPLE_USERS)
    region = random.choice(SAMPLE_REGIONS)
    source_ip = str(ipaddress.IPv4Address(random.randint(1, 0xFFFFFFFF)))
    event = generate_base_event('DescribeInstances', 'ec2.amazonaws.com', region, user, source_ip, is_attack=is_attack)
    event['requestParameters'] = {'maxResults': random.randint(5, 100)}
    # Response elements for DescribeInstances can be complex, keeping it simple/null
    event['responseElements'] = {'requestId': str(uuid.uuid4())}
    return event

def generate_create_user(username=None, is_attack=False):
    """Generates a sample CreateUser event."""
    user = random.choice([u for u in SAMPLE_USERS if u['type'] != 'Root']) # Non-root user performs action
    region = 'us-east-1' # IAM is global, but often logged in a specific region context
    source_ip = str(ipaddress.IPv4Address(random.randint(1, 0xFFFFFFFF)))
    event = generate_base_event('CreateUser', 'iam.amazonaws.com', region, user, source_ip, is_attack=is_attack)
    if not username:
        username = 'generated_user_' + ''.join(random.choices(string.ascii_lowercase, k=6))
    event['requestParameters'] = {'userName': username}
    # Simplified response
    event['responseElements'] = {
        'user': {
            'path': '/',
            'userName': username,
            'userId': 'AIDA' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=16)),
            'arn': f"arn:aws:iam::{user['accountId']}:user/{username}",
            'createDate': event['eventTime']
        }
    }
    event['readOnly'] = False
    return event

# --- Simulation Logic --- 

def generate_log_file(output_path, num_events, attack_ratio=0.1):
    """Generates a log file with a mix of normal and attack events."""
    records = []
    num_attacks = int(num_events * attack_ratio)
    num_normal = num_events - num_attacks

    logging.info(f"Generating {num_events} events ({num_normal} normal, {num_attacks} attack-like) to {output_path}...")

    # Generate normal events
    for _ in range(num_normal):
        event_type = random.choice(['login_success', 'describe_instances'])
        if event_type == 'login_success':
            records.append(generate_console_login(success=True, is_attack=False))
        elif event_type == 'describe_instances':
            records.append(generate_describe_instances(is_attack=False))
        # Add more normal event types
        time.sleep(random.uniform(0.01, 0.1)) # Simulate small time gaps

    # Generate attack-like events (examples)
    for _ in range(num_attacks):
        event_type = random.choice(['login_fail', 'create_user', 'rapid_describe'])
        if event_type == 'login_fail':
            records.append(generate_console_login(success=False, is_attack=True))
        elif event_type == 'create_user':
             # Simulate creating a suspicious user
            suspicious_username = 'suspicious_user_' + ''.join(random.choices(string.ascii_lowercase, k=4))
            records.append(generate_create_user(username=suspicious_username, is_attack=True))
        elif event_type == 'rapid_describe':
            # Simulate rapid recon
            for _ in range(random.randint(3, 7)):
                 # Mark these describe calls as part of an attack sequence
                 records.append(generate_describe_instances(is_attack=True))
                 time.sleep(random.uniform(0.005, 0.05))
        # Add more attack event types
        time.sleep(random.uniform(0.01, 0.1))

    # Shuffle records slightly to mix normal and attack events chronologically
    records.sort(key=lambda x: x['eventTime'])

    # Format for CloudTrail log file
    log_data = {'Records': records}

    # Save to file (optionally compressed)
    try:
        if output_path.endswith('.gz'):
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
        logging.info(f"Successfully generated log file: {output_path}")
    except Exception as e:
        logging.error(f"Failed to write log file {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample CloudTrail-like log files.")
    parser.add_argument('--output-dir', default='../../data/simulated', help="Directory to save generated log files.")
    parser.add_argument('--num-files', type=int, default=1, help="Number of log files to generate.")
    parser.add_argument('--events-per-file', type=int, default=100, help="Number of events per log file.")
    parser.add_argument('--attack-ratio', type=float, default=0.1, help="Approximate ratio of attack-like events (0.0 to 1.0).")
    parser.add_argument('--compress', action='store_true', help="Compress output files with gzip (.json.gz).")

    args = parser.parse_args()

    # Ensure output directory exists
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {args.output_dir}: {e}")
        exit(1)

    for i in range(args.num_files):
        # Generate a filename (e.g., based on timestamp)
        now_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        file_extension = '.json.gz' if args.compress else '.json'
        output_filename = f"sample_cloudtrail_{now_str}_{i:03d}{file_extension}"
        output_path = os.path.join(args.output_dir, output_filename)

        generate_log_file(
            output_path=output_path,
            num_events=args.events_per_file,
            attack_ratio=args.attack_ratio
        )
        # Add a small delay between file generations if needed
        if args.num_files > 1:
            time.sleep(1)

    logging.info("Log generation process complete.")