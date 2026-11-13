"""Opt-in tests against isolated local MySQL and S3-compatible services."""
import asyncio
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import boto3
import mysql.connector
from botocore.exceptions import ClientError

import aws_utils
import db


@unittest.skipUnless(os.getenv('RUN_STORAGE_INTEGRATION') == '1',
                     'Set RUN_STORAGE_INTEGRATION=1 to use isolated local services')
class StorageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        endpoint = os.getenv('TEST_S3_ENDPOINT', 'http://127.0.0.1:19000')
        host = os.getenv('TEST_MYSQL_HOST', '127.0.0.1')
        database = os.getenv('TEST_MYSQL_DATABASE', 'medgemma_integration_test')
        if urlsplit(endpoint).hostname not in {'127.0.0.1', 'localhost', '::1'} or host not in {'127.0.0.1', 'localhost', '::1'}:
            raise RuntimeError('Integration tests are restricted to local services')
        if database != 'medgemma_integration_test':
            raise RuntimeError('Integration tests require the dedicated test database')
        cls.bucket = 'medgemma-integration-test'
        cls.env = patch.dict(os.environ, {
            'MYSQL_HOST': host, 'MYSQL_PORT': os.getenv('TEST_MYSQL_PORT', '13306'),
            'MYSQL_USER': 'medgemma_test', 'MYSQL_PASSWORD': 'local-test-password',
            'MYSQL_DATABASE': database, 'AWS_ENDPOINT_URL_S3': endpoint,
            'AWS_ACCESS_KEY_ID': 'local-test-user', 'AWS_SECRET_ACCESS_KEY': 'local-test-password',
            'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_EC2_METADATA_DISABLED': 'true',
        })
        cls.env.start()
        cls.addClassCleanup(cls.env.stop)
        cls.s3 = boto3.client('s3', region_name='us-east-1')
        deadline = time.monotonic() + 60
        while True:
            try:
                conn = db._connect()
                conn.close()
                cls.s3.list_buckets()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        db.initialize_database()
        try:
            cls.s3.head_bucket(Bucket=cls.bucket)
        except ClientError as exc:
            if exc.response['ResponseMetadata']['HTTPStatusCode'] != 404:
                raise
            cls.s3.create_bucket(Bucket=cls.bucket)

    def test_upload_and_database_record_with_cleanup(self):
        identifier = str(uuid.uuid4())
        key = f'integration-tests/{identifier}/sample.pdf'
        connection = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'sample.pdf'
                payload = b'%PDF-1.4\nsynthetic test data\n'
                path.write_bytes(payload)
                asyncio.run(aws_utils.upload_to_s3(str(path), self.bucket, key, 'us-east-1'))
                asyncio.run(db.store_analysis_result(identifier, key))
            self.assertEqual(self.s3.get_object(Bucket=self.bucket, Key=key)['Body'].read(), payload)
            connection = db._connect()
            cursor = connection.cursor()
            try:
                cursor.execute('SELECT report_path FROM xray_reports WHERE image_uuid = %s', (identifier,))
                self.assertEqual(cursor.fetchone()[0], key)
            finally:
                cursor.close()
        finally:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            if connection is None:
                connection = db._connect()
            try:
                cursor = connection.cursor()
                try:
                    cursor.execute('DELETE FROM xray_reports WHERE image_uuid = %s', (identifier,))
                    connection.commit()
                finally:
                    cursor.close()
            finally:
                connection.close()

    def test_api_report_workflow_with_real_storage_and_mocked_model(self):
        from fastapi.testclient import TestClient
        from PIL import Image
        from src.api.schemas import AnalysisResponse, RadiologyFindings
        import src.api.app as api

        identifier = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        key = f'reports/{analysis_id}/premium_medical_report_{analysis_id}.pdf'
        findings = RadiologyFindings(anatomical_structures=['Chest X-ray'], abnormalities=['Uncertain'],
                                     impression='Indeterminate', recommendations=[], confidence_level='N/A')
        result = AnalysisResponse(analysis_id=analysis_id, timestamp='2026-01-01T00:00:00+00:00',
                                  findings=findings, raw_response='synthetic result', image_metadata={})
        class Analyzer:
            async def analyze_image_from_url(self, url):
                return result, Image.new('RGB', (16, 16)), None

        connection = None
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(api, 'analyzer', Analyzer()), \
                 patch.object(api, 'pdf_generator', api.UltimatePDFGenerator(directory)), \
                 patch.dict(os.environ, {'API_KEY_ADMIN': 'a' * 40, 'API_KEY_ADMIN_FILE': '',
                     'API_KEY_READER': '', 'API_KEY_READER_FILE': '', 'API_KEY_OPERATOR': '',
                     'API_KEY_OPERATOR_FILE': '', 'AWS_S3_BUCKET_NAME': self.bucket, 'AWS_REGION': 'us-east-1'}):
                client = TestClient(api.app, headers={'X-API-Key': 'a' * 40})
                response = client.post('/analyze-url-and-report', json={
                    'image_url': 'https://example.com/synthetic.png', 'image_uuid': identifier})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['pdf_path'], key)
            self.assertTrue(self.s3.get_object(Bucket=self.bucket, Key=key)['Body'].read().startswith(b'%PDF'))
            connection = db._connect()
            cursor = connection.cursor()
            try:
                cursor.execute('SELECT report_path FROM xray_reports WHERE image_uuid = %s', (identifier,))
                self.assertEqual(cursor.fetchone()[0], key)
            finally:
                cursor.close()
        finally:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            if connection is None:
                connection = db._connect()
            try:
                cursor = connection.cursor()
                try:
                    cursor.execute('DELETE FROM xray_reports WHERE image_uuid = %s', (identifier,))
                    connection.commit()
                finally:
                    cursor.close()
            finally:
                connection.close()
