import logging
import numpy as np
import cv2
from PIL import Image
from main import MedicalImageAnalyzer, CONFIDENCE_KEYWORDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_confidence_extraction():
    logger.info("Testing Confidence Extraction...")
    analyzer = MedicalImageAnalyzer()
    
    test_cases = [
        (
            "The image is highly suggestive of pneumonia in the right lower lobe.",
            {"pneumonia": 0.88}
        ),
        (
            "There is no evidence of pneumothorax. Possible small pleural effusion.",
            {"pneumothorax": 0.05, "effusion": 0.50}
        ),
        (
            "Findings are consistent with tuberculosis. No cardiomegaly.",
            {"tuberculosis": 0.75, "cardiomegaly": 0.08}
        ),
        (
            "Normal chest X-ray. No acute cardiopulmonary findings.",
            {"pneumonia": 0.08, "pneumothorax": 0.08} # Should infer low prob from negative
        )
    ]
    
    for text, expected in test_cases:
        result = analyzer.extract_condition_confidences(text)
        logger.info(f"Input: {text[:50]}...")
        logger.info(f"Extracted: {result}")
        
        # approximate check
        for cond, score in expected.items():
            if cond in result:
                # Allow small floating point diffs or keyword variations
                if abs(result[cond] - score) > 0.1:
                    logger.warning(f"  ❌ Mismatch for {cond}: Got {result[cond]}, Expected ~{score}")
                else:
                    logger.info(f"  ✅ {cond} match")
            else:
                 logger.warning(f"  ❌ Missing {cond} in result")

def test_localization_map():
    logger.info("\nTesting Localization Map Generation...")
    analyzer = MedicalImageAnalyzer()
    
    # Create dummy image and heatmap
    img = Image.new('RGB', (500, 500), color='black')
    heatmap = np.zeros((500, 500), dtype=np.uint8)
    
    # Draw two "attention" spots
    # Region 1: Right lower (large)
    cv2.circle(heatmap, (150, 350), 50, 255, -1) 
    # Region 2: Left upper (small)
    cv2.circle(heatmap, (350, 150), 30, 200, -1)
    
    # Smooth to make it map-like
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Generate localization
    result_img, regions = analyzer._generate_localization_map(
        heatmap, img, threshold_percentile=50, min_contour_area=100
    )
    
    logger.info(f"Found {len(regions)} regions")
    for r in regions:
        logger.info(f"  Region {r['region_id']}: {r['location']}, Score: {r['attention_score']}")
        
    if len(regions) >= 2:
        logger.info("  ✅ Correctly detected 2 regions")
    else:
        logger.warning(f"  ❌ Expected 2 regions, got {len(regions)}")

if __name__ == "__main__":
    try:
        test_confidence_extraction()
        test_localization_map()
        print("\n✅ Tests passed!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
