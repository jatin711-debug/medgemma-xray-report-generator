import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.api.app as api
from src.api.security import Role, configured_keys


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        self.keys = {'API_KEY_ADMIN': 'a' * 40, 'API_KEY_OPERATOR': 'o' * 40,
                     'API_KEY_READER': 'r' * 40, 'API_KEY_ADMIN_FILE': '',
                     'API_KEY_OPERATOR_FILE': '', 'API_KEY_READER_FILE': ''}
        self.environment = patch.dict('os.environ', self.keys)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = TestClient(api.app)

    def test_liveness_is_public_but_data_routes_require_a_key(self):
        self.assertEqual(self.client.get('/health/live').status_code, 200)
        self.assertEqual(self.client.get('/health/ready').status_code, 401)
        self.assertEqual(self.client.get('/reports').status_code, 401)
        self.assertEqual(self.client.get('/reports', headers={'X-API-Key': 'wrong'}).status_code, 401)

    def test_readiness_reports_model_state(self):
        class Analyzer:
            model_loaded = False
        with patch.object(api, 'analyzer', Analyzer()):
            headers = {'X-API-Key': self.keys['API_KEY_READER']}
            self.assertEqual(self.client.get('/health/ready', headers=headers).status_code, 503)
            api.analyzer.model_loaded = True
            self.assertEqual(self.client.get('/health/ready', headers=headers).status_code, 200)

    def test_reader_operator_and_admin_permissions(self):
        reader = {'X-API-Key': self.keys['API_KEY_READER']}
        operator = {'X-API-Key': self.keys['API_KEY_OPERATOR']}
        admin = {'X-API-Key': self.keys['API_KEY_ADMIN']}
        self.assertEqual(self.client.get('/reports', headers=reader).status_code, 200)
        request = {'image_url': 'https://example.com/image.png'}
        self.assertEqual(self.client.post('/analyze-url', headers=reader, json=request).status_code, 403)
        self.assertEqual(self.client.post('/load-model', headers=operator).status_code, 403)
        with patch.object(api.analyzer, 'analyze_image_from_url', new_callable=AsyncMock) as analyze:
            analyze.side_effect = HTTPException(status_code=418, detail='Analyzer reached')
            self.assertEqual(self.client.post('/analyze-url', headers=operator, json=request).status_code, 418)
            self.assertEqual(self.client.post('/analyze-url', headers=admin, json=request).status_code, 418)

    def test_only_admin_can_reload_model(self):
        class Analyzer:
            reload_model = AsyncMock()
        with patch.object(api, 'analyzer', Analyzer()):
            response = self.client.post('/load-model', headers={'X-API-Key': self.keys['API_KEY_ADMIN']})
        self.assertEqual(response.status_code, 200)
        Analyzer.reload_model.assert_awaited_once()

    def test_startup_rejects_missing_admin_before_loading_model(self):
        with patch.dict('os.environ', {'API_KEY_ADMIN': ''}), \
             patch.object(api.MedicalImageAnalyzer, 'load_model', new_callable=AsyncMock) as load:
            with self.assertRaises(ValueError):
                with TestClient(api.app):
                    pass
            load.assert_not_awaited()

    def test_missing_invalid_or_duplicate_configuration_fails_closed(self):
        with patch.dict('os.environ', {'API_KEY_ADMIN': ''}):
            self.assertEqual(self.client.get('/reports', headers={'X-API-Key': 'a' * 40}).status_code, 503)
        with patch.dict('os.environ', {'API_KEY_ADMIN': 'short'}):
            with self.assertRaises(ValueError): configured_keys()
        with patch.dict('os.environ', {'API_KEY_READER': 'a' * 40}):
            with self.assertRaises(ValueError): configured_keys()

    def test_admin_key_may_be_loaded_from_a_secret_file(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / 'key'
            key_file.write_text('s' * 40, encoding='utf-8')
            with patch.dict('os.environ', {'API_KEY_ADMIN': '', 'API_KEY_ADMIN_FILE': str(key_file)}):
                self.assertEqual(configured_keys()[Role.ADMIN], b's' * 40)
                self.assertEqual(self.client.get('/reports', headers={'X-API-Key': 's' * 40}).status_code, 200)
