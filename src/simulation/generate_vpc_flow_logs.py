# Simulates AWS VPC Flow Logs

import random
import time
import datetime
import gzip
import os
import argparse
import logging
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [VPC_SIM] - %(message)s')

fake = Faker()

# --- Configuration ---
DEFAULT_OUTPUT_DIR = '../../data/raw_logs' # Place alongside CloudTrail logs
DEFAULT_NUM_FILES = 1
DEFAULT_EVENTS_PER_FILE = 100
DEFAULT_VERSION = 2
DEFAULT_ACCOUNT_ID = '123456789012'
DEFAULT_INTERFACE_ID_PREFIX = 'eni-'
DEFAULT_ACTION = ['ACCEPT', 'REJECT']
DEFAULT_LOG_STATUS = ['OK', 'NODATA', 'SKIPDATA']
COMMON_PORTS = [80, 443, 22, 53, 123, 3389]
PROTOCOLS = {'tcp': 6, 'udp': 17, 'icmp': 1}

# --- Helper Functions ---

def generate_ip():
    """Generates a random private or public IP address."""
    if random.random() < 0.7: # Higher chance of private IPs
        # Simplified private ranges
        block = random.choice(['10.', '172.16.', '192.168.'])
        if block == '10.':
            return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        elif block == '172.16.':
            return f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
    else:
        return fake.ipv4()

def generate_port():
    """Generates a random port, favoring common ports."""
    if random.random() < 0.5:
        return random.choice(COMMON_PORTS)
    else:
        return random.randint(1025, 65535)

def generate_interface_id():
    """Generates a random ENI-like ID."""
    return DEFAULT_INTERFACE_ID_PREFIX + ''.join(random.choices('0123456789abcdef', k=17))

def generate_flow_log_record(version=DEFAULT_VERSION, account_id=DEFAULT_ACCOUNT_ID, is_attack=False):
    """Generates a single VPC Flow Log record string and the attack flag."""
    interface_id = generate_interface_id()
    srcaddr = generate_ip()
    dstaddr = generate_ip()
    srcport = generate_port()
    dstport = generate_port()
    protocol_name = random.choice(list(PROTOCOLS.keys()))
    protocol = PROTOCOLS[protocol_name]
    packets = random.randint(1, 100)
    bytes_val = packets * random.randint(60, 1500)
    # Simulate short time window for the flow
    end_time = int(time.time()) - random.randint(1, 60)
    start_time = end_time - random.randint(1, 300) # Flow duration up to 5 mins
    action = random.choice(DEFAULT_ACTION)
    log_status = random.choice(DEFAULT_LOG_STATUS)

    # --- Basic Attack Simulation Placeholder ---
    # Keep track if this specific record was modified to be an attack
    record_is_attack = False
    if is_attack:
        # Example: Simulate port scanning (many connections to different ports)
        if random.random() < 0.3:
            dstport = random.randint(1, 1024) # Scan common low ports
            action = 'REJECT' # Often rejected by firewall/SG
            packets = 1
            bytes_val = random.randint(40, 100)
            record_is_attack = True # Mark this record as an attack
        # Example: Simulate large data transfer (high byte count)
        elif random.random() < 0.2:
            bytes_val = random.randint(10_000_000, 100_000_000) # 10MB - 100MB
            action = 'ACCEPT'
            record_is_attack = True # Mark this record as an attack
        # Add more attack patterns here...

    # Handle ICMP (no ports)
    if protocol == 1:
        srcport = '-'
        dstport = '-'

    record = [
        str(version),
        account_id,
        interface_id,
        srcaddr,
        dstaddr,
        str(srcport),
        str(dstport),
        str(protocol),
        str(packets),
        str(bytes_val),
        str(start_time),
        str(end_time),
        action,
        log_status
    ]
    # Return the record string and the boolean attack flag for this record
    return ' '.join(record), record_is_attack

def generate_log_file(filename, num_events, attack_ratio=0.1):
    """Generates a single VPC Flow Log file (gzipped), including the attack label."""
    logging.info(f"Generating file: {filename} with {num_events} events (Attack Ratio: {attack_ratio:.2f})")
    records_with_labels = []
    num_attacks_to_generate = int(num_events * attack_ratio)
    generated_attacks = 0

    # Generate header - adding the label column
    header = "version account-id interface-id srcaddr dstaddr srcport dstport protocol packets bytes start end action log-status is_simulated_attack"
    records_with_labels.append(header)

    # Generate events, aiming for the target attack ratio
    # We generate slightly more than num_events initially to ensure enough attacks are created
    # by the probabilistic attack logic inside generate_flow_log_record
    target_records = []
    while len(target_records) < num_events:
        is_potential_attack = (generated_attacks < num_attacks_to_generate) and (random.random() < attack_ratio * 1.5) # Increase chance if below target
        record_str, record_is_attack = generate_flow_log_record(is_attack=is_potential_attack)

        # Only count as an attack if the generation logic actually made it one
        if record_is_attack:
            if generated_attacks < num_attacks_to_generate:
                target_records.append(f"{record_str} 1") # Append attack label 1
                generated_attacks += 1
            else:
                # If we already have enough attacks, generate a normal record instead
                record_str_normal, _ = generate_flow_log_record(is_attack=False)
                target_records.append(f"{record_str_normal} 0") # Append normal label 0
        else:
            # If it wasn't flagged as an attack by generation logic, or we needed a normal one
            target_records.append(f"{record_str} 0") # Append normal label 0

        # Ensure we don't exceed the total number of events
        if len(target_records) >= num_events:
            break

    # Shuffle records (excluding header)
    header = records_with_labels[0]
    # Take exactly num_events records and shuffle them
    content_records = target_records[:num_events]
    random.shuffle(content_records)
    final_records = [header] + content_records

    try:
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            for record in final_records:
                f.write(record + '\n')
        logging.info(f"Successfully generated {filename} (Attacks generated: {generated_attacks})")
    except Exception as e:
        logging.error(f"Error writing to file {filename}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated AWS VPC Flow Log files.")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the generated log files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('--num-files', type=int, default=DEFAULT_NUM_FILES,
                        help=f"Number of log files to generate (default: {DEFAULT_NUM_FILES})")
    parser.add_argument('--events-per-file', type=int, default=DEFAULT_EVENTS_PER_FILE,
                        help=f"Number of events per log file (default: {DEFAULT_EVENTS_PER_FILE})")
    parser.add_argument('--attack-ratio', type=float, default=0.1,
                        help="Approximate ratio of attack events to generate (0.0 to 1.0, default: 0.1)")
    parser.add_argument('--compress', action='store_true', default=True,
                        help="Compress output files with gzip (default: True)")
    parser.add_argument('--no-compress', dest='compress', action='store_false',
                        help="Do not compress output files.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Starting VPC Flow Log generation...")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Number of Files: {args.num_files}")
    logging.info(f"Events per File: {args.events_per_file}")
    logging.info(f"Attack Ratio: {args.attack_ratio}")
    logging.info(f"Compression Enabled: {args.compress}")

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")

    for i in range(args.num_files):
        file_suffix = f"{timestamp}_{i:03d}.log"
        if args.compress:
            file_suffix += '.gz'
        filename = os.path.join(args.output_dir, f"vpc_flow_logs_{file_suffix}")
        generate_log_file(filename, args.events_per_file, args.attack_ratio)

    logging.info("VPC Flow Log generation finished.")