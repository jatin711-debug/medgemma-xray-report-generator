import asyncio
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import src.api.app as api
from src.api.schemas import AnalysisResponse, RadiologyFindings
from src.core.medical_analyzer import MedicalImageAnalyzer


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {'API_KEY_ADMIN': 'a' * 40,
            'API_KEY_READER': '', 'API_KEY_OPERATOR': '', 'API_KEY_ADMIN_FILE': '',
            'API_KEY_READER_FILE': '', 'API_KEY_OPERATOR_FILE': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_full_precision_option_disables_quantization(self):
        analyzer = MedicalImageAnalyzer()
        with patch.dict('os.environ', {'MEDGEMMA_QUANTIZE': '0'}), \
             patch('src.core.medical_analyzer.torch.cuda.is_available', return_value=True), \
             patch.object(analyzer, 'get_system_memory_info', return_value='test'), \
             patch('src.core.medical_analyzer.AutoProcessor.from_pretrained'), \
             patch('src.core.medical_analyzer.AutoModelForImageTextToText.from_pretrained') as load, \
             patch('src.core.medical_analyzer.BitsAndBytesConfig') as quantization:
            asyncio.run(analyzer.load_model())
        self.assertTrue(analyzer.model_loaded)
        self.assertIsNone(load.call_args.kwargs['quantization_config'])
        quantization.assert_not_called()

    def test_inference_is_serialized_and_does_not_block_event_loop(self):
        analyzer = MedicalImageAnalyzer()
        active = 0
        peak = 0
        thread_guard = threading.Lock()
        def fake_inference(image):
            nonlocal active, peak
            with thread_guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.08)
            with thread_guard:
                active -= 1
            return 'raw', None
        async def exercise():
            image = Image.new('RGB', (4, 4))
            tasks = [asyncio.create_task(analyzer.analyze_image(image)) for _ in range(2)]
            await asyncio.sleep(0.02)
            self.assertFalse(all(task.done() for task in tasks))
            await asyncio.gather(*tasks)
        with patch.object(analyzer, '_load_model'), patch.object(analyzer, '_analyze_image', fake_inference):
            asyncio.run(exercise())
        self.assertEqual(peak, 1)

    def test_upload_endpoint_passes_analysis_to_report(self):
        findings = RadiologyFindings(anatomical_structures=[], abnormalities=[],
                                     impression='Indeterminate', recommendations=[], confidence_level='N/A')
        result = AnalysisResponse(analysis_id='test-id', timestamp='2026-01-01',
                                  findings=findings, raw_response='uncertain', image_metadata={})
        image = Image.new('RGB', (4, 4))

        class Analyzer:
            async def analyze_image_from_upload(self, file):
                return result, image

        class Generator:
            def generate_pdf_report(self, response, original):
                assert response.analysis_id == 'test-id' and original is image
                return 'reports/test.pdf'

        with patch.object(api, 'analyzer', Analyzer()), patch.object(api, 'pdf_generator', Generator()):
            client = TestClient(api.app, headers={'X-API-Key': 'a' * 40})
            response = client.post('/analyze-and-report', files={'file': ('x.png', b'content', 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'pdf_path': 'reports/test.pdf', 'analysis_id': 'test-id'})

    def test_report_generation_with_indeterminate_output(self):
        findings = RadiologyFindings(anatomical_structures=['Chest X-ray'],
                                     abnormalities=['Uncertain result'], impression='Indeterminate',
                                     recommendations=['Research only'], confidence_level='Not available')
        result = AnalysisResponse(analysis_id='test-id', timestamp='2026-01-01', findings=findings,
                                  raw_response='Uncertain <finding>', image_metadata={'filename': '<scan & input>.png'})
        with tempfile.TemporaryDirectory() as directory:
            generator = api.UltimatePDFGenerator(directory)
            path = generator.generate_pdf_report(result, Image.new('RGB', (64, 64)))
            self.assertTrue(Path(path).read_bytes().startswith(b'%PDF'))

    def test_report_list_uses_correct_analysis_id(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'premium_medical_report_test-id.pdf').write_bytes(b'%PDF')
            class Generator:
                reports_dir = directory
            with patch.object(api, 'pdf_generator', Generator()):
                response = TestClient(api.app, headers={'X-API-Key': 'a' * 40}).get('/reports')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['reports'][0]['analysis_id'], 'test-id')

    def test_s3_configuration_checked_before_inference(self):
        class Analyzer:
            async def analyze_image_from_url(self, url):
                raise AssertionError('inference should not run')
        with patch.object(api, 'analyzer', Analyzer()), \
             patch.dict('os.environ', {'AWS_S3_BUCKET_NAME': '', 'AWS_REGION': ''}):
            response = TestClient(api.app, headers={'X-API-Key': 'a' * 40}).post('/analyze-url-and-report', json={
                'image_url': 'https://example.com/image.png', 'image_uuid': 'sample'})
        self.assertEqual(response.status_code, 503)

    def test_url_fetch_disabled_by_default(self):
        with patch.dict('os.environ', {'ENABLE_URL_FETCH': ''}):
            client = TestClient(api.app, headers={'X-API-Key': 'a' * 40})
            response = client.post('/analyze-url', json={'image_url': 'https://example.com/x.png'})
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
