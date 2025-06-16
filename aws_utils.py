\
import asyncio
import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv(dotenv_path=".env") if os.path.exists(".env") else None
# Ensure that the environment variables for AWS credentials are set
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# Configure logging
logger = logging.getLogger(__name__)
# Ensure basicConfig is called only if no handlers are configured
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Thread pool executor for running blocking I/O (boto3) in async context
# Define it at the module level so it's reused
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

async def upload_to_s3(local_file_path: str, bucket_name: str, s3_key: str, region_name: str = "us-east-1") -> str:
    """
    Uploads a file to an S3 bucket and returns the S3 object URL.

    Args:
        local_file_path: The local path to the file to upload.
        bucket_name: The name of the S3 bucket.
        s3_key: The key (path/filename) for the object in S3.
        region_name: The AWS region of the bucket. Credentials should be configured
                     in the environment (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
                     or via IAM roles if running on AWS infrastructure.

    Returns:
        The HTTPS URL of the uploaded object in S3.

    Raises:
        FileNotFoundError: If the local file does not exist.
        Exception: If the upload fails due to AWS issues (credentials, permissions, etc.)
                   or other unexpected errors.
    """
    if not os.path.exists(local_file_path):
        logger.error(f"Local file not found for S3 upload: {local_file_path}")
        raise FileNotFoundError(f"Local file not found: {local_file_path}")

    loop = asyncio.get_event_loop()
    
    # boto3.client is not thread-safe for all operations if the client object itself is shared across threads
    # without proper locking. However, creating it per call or ensuring it's used by one thread at a time
    # from the executor is generally safe. For simplicity here, it's created before the executor call.
    # For high-concurrency, consider passing client factory or managing client lifecycle carefully.
    s3_client = boto3.client('s3', region_name=region_name)

    try:
        logger.info(f"Starting S3 upload of {local_file_path} to s3://{bucket_name}/{s3_key}")
        await loop.run_in_executor(
            executor,
            s3_client.upload_file,
            local_file_path,
            bucket_name,
            s3_key
        )
        
        # Construct the S3 HTTPS URL
        s3_object_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"

        logger.info(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
        logger.info(f"S3 URL: {s3_object_url}")
        return s3_object_url
    except FileNotFoundError:
        logger.error(f"Upload failed: File not found at {local_file_path} (should have been caught earlier)")
        raise # Re-raise the specific FileNotFoundError
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.error(f"AWS credentials not found or incomplete for S3 upload: {e}")
        raise Exception("AWS credentials not configured properly for S3 upload.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        logger.error(f"AWS S3 ClientError during upload to s3://{bucket_name}/{s3_key}. Code: {error_code}, Message: {error_message}. Details: {e}")
        raise Exception(f"S3 upload failed: {error_code} - {error_message}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during S3 upload of {local_file_path} to s3://{bucket_name}/{s3_key}: {e}", exc_info=True)
        raise Exception(f"S3 upload failed due to an unexpected error: {e}")

# Example of how to gracefully shutdown the executor if the application has a clear shutdown hook
# This is more relevant for long-running applications.
# def shutdown_executor():
#     logger.info("Shutting down ThreadPoolExecutor for S3 uploads.")
#     executor.shutdown(wait=True)
#
# If using FastAPI, you might tie this to shutdown events:
# app.add_event_handler("shutdown", shutdown_executor)
