"""MySQL storage helpers for the optional S3 reporting workflow."""
import logging
import os

import mysql.connector
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


def _connect():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"), user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"), database=os.getenv("MYSQL_DATABASE"),
        port=int(os.getenv("MYSQL_PORT", "3306")), connection_timeout=10,
    )


def _store_analysis_result(image_uuid: str, report_path: str) -> None:
    connection = None
    cursor = None
    try:
        connection = _connect()
        cursor = connection.cursor()
        cursor.execute("INSERT INTO xray_reports (image_uuid, report_path) VALUES (%s, %s)",
                       (image_uuid, report_path))
        connection.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


async def store_analysis_result(image_uuid: str, reportPath: str) -> None:
    """Store a report key without blocking the server's event loop."""
    try:
        await run_in_threadpool(_store_analysis_result, image_uuid, reportPath)
    except (mysql.connector.Error, ValueError) as exc:
        logger.exception("Failed to store report metadata")
        raise HTTPException(status_code=500, detail="Failed to store report metadata") from exc


def initialize_database() -> None:
    """Create the report table if it is missing; run explicitly during setup."""
    connection = None
    cursor = None
    try:
        connection = _connect()
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS xray_reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_uuid VARCHAR(255) NOT NULL UNIQUE,
            report_path VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )""")
        connection.commit()
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
