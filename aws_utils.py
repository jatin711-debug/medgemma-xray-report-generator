"""S3 upload helper for the optional report-storage integration."""
from pathlib import Path

import boto3
from fastapi.concurrency import run_in_threadpool


def _upload(local_file_path: str, bucket_name: str, s3_key: str, region_name: str) -> None:
    client = boto3.client("s3", region_name=region_name)
    client.upload_file(local_file_path, bucket_name, s3_key)


async def upload_to_s3(local_file_path: str, bucket_name: str, s3_key: str,
                       region_name: str = "us-east-1") -> str:
    """Upload a local report in a worker thread and return its S3 URI."""
    if not Path(local_file_path).is_file():
        raise FileNotFoundError("Local report does not exist")
    if not bucket_name or not region_name:
        raise ValueError("S3 bucket and region are required")
    await run_in_threadpool(_upload, local_file_path, bucket_name, s3_key, region_name)
    return f"s3://{bucket_name}/{s3_key}"
