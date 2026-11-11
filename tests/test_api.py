import asyncio
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import src.api.app as api
from src.api.schemas import AnalysisResponse, RadiologyFindings
from src.core.medical_analyzer import MedicalImageAnalyzer


class AnalysisTests(unittest.TestCase):
    def test_unrecognized_output_is_indeterminate(self):
        analyzer = MedicalImageAnalyzer()
        for raw in ('uncertain', '**Finding**: possible pneumothorax'):
            self.assertEqual(analyzer.parse_ai_response(raw).impression, 'Indeterminate')
        self.assertEqual(analyzer.parse_ai_response('**Finding**: No pneumothorax detected').impression,
                         'No Pneumothorax Detected')
        self.assertEqual(analyzer.parse_ai_response('**Finding**: Pneumothorax detected').impression,
                         'Pneumothorax Detected')

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
            client = TestClient(api.app)
            response = client.post('/analyze-and-report', files={'file': ('x.png', b'content', 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'pdf_path': 'reports/test.pdf', 'analysis_id': 'test-id'})

    def test_report_generation_with_indeterminate_output(self):
        findings = RadiologyFindings(anatomical_structures=['Chest X-ray'],
                                     abnormalities=['Uncertain result'], impression='Indeterminate',
                                     recommendations=['Research only'], confidence_level='Not available')
        result = AnalysisResponse(analysis_id='test-id', timestamp='2026-01-01', findings=findings,
                                  raw_response='Uncertain <finding>', image_metadata={})
        with tempfile.TemporaryDirectory() as directory:
            generator = api.UltimatePDFGenerator(directory)
            path = generator.generate_pdf_report(result, Image.new('RGB', (64, 64)))
            self.assertTrue(Path(path).read_bytes().startswith(b'%PDF'))

    def test_url_fetch_disabled_by_default(self):
        with patch.dict('os.environ', {'ENABLE_URL_FETCH': ''}):
            client = TestClient(api.app)
            response = client.post('/analyze-url', json={'image_url': 'https://example.com/x.png'})
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
