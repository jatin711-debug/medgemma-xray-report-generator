"""Optional smoke test; not a model-quality or clinical evaluation."""
import os
import tempfile
import unittest
from pathlib import Path
from fastapi import UploadFile
from src.core.medical_analyzer import MedicalImageAnalyzer
from src.reports.generator import UltimatePDFGenerator

@unittest.skipUnless(os.getenv('RUN_MODEL_SMOKE') == '1', 'Model smoke test is opt-in')
class ModelSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_model_on_explicit_sample_image(self):
        path_value = os.getenv('MODEL_SMOKE_IMAGE')
        if not path_value:
            self.fail('Set MODEL_SMOKE_IMAGE to a non-sensitive sample image')
        path = Path(path_value)
        analyzer = MedicalImageAnalyzer()
        await analyzer.load_model()
        with path.open('rb') as image_file:
            result, image, attribution = await analyzer.analyze_image_from_upload(
                UploadFile(filename=path.name, file=image_file))
        self.assertTrue(result.raw_response.strip())
        self.assertTrue(result.analysis_id)
        self.assertGreater(image.width, 0)
        self.assertIn('generation_seconds', result.timings)
        # A shipped calibration must match the default model, prompt and the precision in use.
        self.assertIn(result.decision.get('band'), {'detected', 'review', 'negative'},
                      result.decision.get('reason'))
        self.assertEqual(result.findings.impression, result.decision['outcome'])
        self.assertIn(result.attribution.get('status'), {'available', 'unavailable', 'disabled'})
        if attribution is not None:
            side = attribution.grid.shape[0]
            self.assertEqual(attribution.grid.shape, (side, side))
            self.assertGreaterEqual(float(attribution.grid.min()), -1.0)
            self.assertLessEqual(float(attribution.grid.max()), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            report = UltimatePDFGenerator(directory).generate_pdf_report(result, image, attribution)
            self.assertTrue(Path(report).read_bytes().startswith(b'%PDF'))
