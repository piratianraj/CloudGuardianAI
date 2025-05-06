# Ingests CloudTrail logs from an S3 bucket

import boto3
import os
import logging
import argparse
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [INGESTION] - %(message)s')

def download_logs_from_s3(bucket_name, prefix, local_dir, region_name='us-east-1', max_files=None):
    """Downloads CloudTrail logs (.json.gz) from S3 to a local directory."""
    logging.info(f"Starting log ingestion from s3://{bucket_name}/{prefix} to {local_dir}")

    # Ensure local directory exists
    try:
        os.makedirs(local_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create local directory {local_dir}: {e}")
        return

    # Initialize Boto3 S3 client
    try:
        session = boto3.Session(region_name=region_name)
        s3 = session.client('s3')
        logging.info(f"Boto3 S3 client initialized in region: {session.region_name}")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"AWS credentials not found or incomplete: {e}")
        logging.error("Please configure your AWS credentials (e.g., via environment variables, ~/.aws/credentials, or IAM role).")
        return
    except Exception as e:
        logging.error(f"Failed to initialize Boto3 session: {e}")
        return

    paginator = s3.get_paginator('list_objects_v2')
    files_downloaded = 0
    total_files_processed = 0

    try:
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    total_files_processed += 1

                    # Filter for CloudTrail log files (typically .json.gz)
                    if not s3_key.endswith('.json.gz'):
                        logging.debug(f"Skipping non-log file: {s3_key}")
                        continue

                    # Construct local file path, preserving directory structure under prefix
                    relative_path = os.path.relpath(s3_key, prefix)
                    local_file_path = os.path.join(local_dir, relative_path)
                    local_file_dir = os.path.dirname(local_file_path)

                    # Ensure local subdirectory exists
                    if not os.path.exists(local_file_dir):
                        try:
                            os.makedirs(local_file_dir, exist_ok=True)
                        except OSError as e:
                            logging.error(f"Failed to create subdirectory {local_file_dir}: {e}")
                            continue # Skip this file

                    # Check if file already exists locally (optional, prevents re-downloading)
                    if os.path.exists(local_file_path):
                        logging.debug(f"File already exists locally, skipping: {local_file_path}")
                        continue

                    # Download the file
                    try:
                        logging.info(f"Downloading {s3_key} to {local_file_path}...")
                        s3.download_file(bucket_name, s3_key, local_file_path)
                        files_downloaded += 1
                        if max_files is not None and files_downloaded >= max_files:
                            logging.info(f"Reached maximum download limit ({max_files} files). Stopping.")
                            break # Stop downloading
                    except ClientError as e:
                        logging.error(f"Failed to download {s3_key}: {e}")
                    except Exception as e:
                        logging.error(f"An unexpected error occurred downloading {s3_key}: {e}")
            else:
                logging.info("No objects found in the specified prefix.")

            if max_files is not None and files_downloaded >= max_files:
                break # Stop iterating through pages

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            logging.error(f"S3 bucket '{bucket_name}' not found or access denied.")
        else:
            logging.error(f"An S3 client error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred listing S3 objects: {e}")

    logging.info(f"Log ingestion finished. Processed {total_files_processed} S3 objects. Downloaded {files_downloaded} new log files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CloudTrail logs from S3.")
    parser.add_argument('--bucket', required=True, help="S3 bucket name containing CloudTrail logs.")
    parser.add_argument('--prefix', required=True, help="S3 prefix for CloudTrail logs (e.g., 'AWSLogs/123456789012/CloudTrail/us-east-1/').")
    parser.add_argument('--local-dir', default='../../data/raw_logs', help="Local directory to download logs into.")
    parser.add_argument('--region', default='us-east-1', help="AWS region for the S3 bucket.")
    parser.add_argument('--max-files', type=int, default=None, help="Maximum number of log files to download (optional).")

    args = parser.parse_args()

    # Example usage from command line:
    # python ingest_logs.py --bucket your-cloudtrail-bucket --prefix AWSLogs/123.../CloudTrail/us-east-1/ --region us-east-1 --max-files 100

    download_logs_from_s3(
        bucket_name=args.bucket,
        prefix=args.prefix,
        local_dir=args.local_dir,
        region_name=args.region,
        max_files=args.max_files
    )