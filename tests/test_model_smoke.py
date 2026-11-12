"""Optional smoke test; not a model-quality or clinical evaluation."""
import os
import unittest
from pathlib import Path
from fastapi import UploadFile
from src.core.medical_analyzer import MedicalImageAnalyzer

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
            result, image = await analyzer.analyze_image_from_upload(
                UploadFile(filename=path.name, file=image_file))
        self.assertTrue(result.raw_response.strip())
        self.assertTrue(result.analysis_id)
        self.assertGreater(image.width, 0)
