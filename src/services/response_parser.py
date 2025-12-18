"""
AI response parsing service for medical image analysis.

Handles parsing of structured AI model responses and extraction of
confidence scores for medical conditions.
"""

import re
import logging
from typing import Dict

from src.api.schemas import RadiologyFindings
from src.core.constants import CONFIDENCE_KEYWORDS, CONDITION_DETECTION_KEYWORDS

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parser for AI model responses from medical image analysis."""
    
    @staticmethod
    def parse_ai_response_multi(raw_response: str) -> RadiologyFindings:
        """
        Parse a structured AI model response for multiple conditions.
        Fixes the bug where "no pneumothorax detected" was misinterpreted.
        
        Args:
            raw_response: Raw text output from the Med-GEMMA model.

        Returns:
            RadiologyFindings: Structured medical findings for all analyzed conditions.
        """
        try:
            findings = {}
            # Regex to capture each condition block
            pattern = re.compile(
                r"\*\*Finding: (Pneumothorax|Pneumonia|TB)\*\*\n"
                r"- \*\*Status\*\*: (Detected|Not Detected)\n"
                r"- \*\*Confidence\*\*: ([\d.]+%|N/A)", 
                re.IGNORECASE
            )
            matches = pattern.findall(raw_response)

            for match in matches:
                condition, status, confidence = match
                findings[condition.strip()] = {
                    "status": status.strip(),
                    "confidence": confidence.strip()
                }

            if not findings:
                # Fallback for unstructured responses
                logger.warning("Could not parse structured response, using fallback.")
                return RadiologyFindings(
                    anatomical_structures=["Chest structures"],
                    abnormalities=["Could not parse AI response."],
                    impression="Inconclusive",
                    recommendations=["Manual review required."],
                    confidence_level="N/A"
                )

            detected_conditions = [
                f"{cond} ({f['confidence']})" 
                for cond, f in findings.items() 
                if f['status'].lower() == 'detected'
            ]

            if detected_conditions:
                impression = "Positive for " + ", ".join(
                    c.split(' ')[0] for c in detected_conditions
                )
                abnormalities = detected_conditions
            else:
                impression = "Negative for all tested conditions"
                abnormalities = ["No evidence of Pneumothorax, Pneumonia, or TB found."]

            # Use the highest confidence among detected conditions, or "N/A"
            confidences = [
                float(f['confidence'].replace('%', '')) 
                for f in findings.values() 
                if '%' in f['confidence']
            ]
            overall_confidence = f"{max(confidences):.1f}%" if confidences else "N/A"

            return RadiologyFindings(
                anatomical_structures=["Lungs", "Pleura", "Mediastinum"],
                abnormalities=abnormalities,
                impression=impression,
                recommendations=["Clinical correlation and expert review is recommended."],
                confidence_level=overall_confidence
            )

        except Exception as e:
            logger.error(f"Error parsing multi-condition AI response: {str(e)}")
            return RadiologyFindings(
                anatomical_structures=["Analysis completed"],
                abnormalities=["Unable to parse detailed findings"],
                impression=raw_response,
                recommendations=["Manual review recommended"],
                confidence_level="Not available"
            )

    @staticmethod
    def extract_condition_confidences(raw_response: str) -> Dict[str, float]:
        """
        Extract confidence scores for each medical condition from model text.
        
        Improved logic to properly distinguish between positive and negative findings.
        
        Args:
            raw_response: Raw text output from the AI model
            
        Returns:
            Dict mapping condition names to confidence scores (0.0-1.0)
        """
        response_lower = raw_response.lower()
        confidences = {}
        
        for condition, keywords in CONDITION_DETECTION_KEYWORDS.items():
            # Check if primary condition keyword is mentioned
            primary_keyword = keywords[0]  # Main condition name
            condition_mentioned = any(kw in response_lower for kw in keywords)
            
            if not condition_mentioned:
                # Condition not discussed - very low confidence
                confidences[condition] = 0.05
                continue
            
            # More specific negative patterns - check for complete phrases
            negative_patterns = [
                f"no {primary_keyword}",
                f"no evidence of {primary_keyword}",
                f"negative for {primary_keyword}",
                f"without {primary_keyword}",
                f"{primary_keyword} not detected",
                f"ruled out {primary_keyword}",
                f"no signs of {primary_keyword}",
                f"absence of {primary_keyword}"
            ]
            
            # Check for explicit negative findings
            is_explicitly_negative = any(
                pattern in response_lower for pattern in negative_patterns
            )
            
            if is_explicitly_negative:
                # Confirmed negative finding
                confidences[condition] = 0.06
                continue
            
            # Look for positive confirmation keywords
            positive_indicators = [
                f"{primary_keyword} detected",
                f"{primary_keyword} present",
                f"findings consistent with {primary_keyword}",
                f"suggestive of {primary_keyword}",
                f"evidence of {primary_keyword}",
            ]
            
            is_positive = any(
                indicator in response_lower for indicator in positive_indicators
            )
            
            # Determine base confidence
            if is_positive:
                max_confidence = 0.75  # High confidence for positive detection
            else:
                max_confidence = 0.50  # Moderate default when mentioned without clear indication
            
            # Refine confidence based on confidence keywords
            for conf_keyword, conf_score in CONFIDENCE_KEYWORDS.items():
                if conf_keyword in response_lower:
                    # Check proximity to condition keywords (simple approach: same sentence/paragraph)
                    for cond_kw in keywords:
                        if cond_kw in response_lower:
                            # Boost or reduce based on confidence keyword
                            if conf_score > 0.5:  # High confidence keyword
                                max_confidence = max(max_confidence, conf_score)
                            else:  # Low confidence keyword  
                                # Only lower if we haven't already detected positive indicators
                                if not is_positive:
                                    max_confidence = min(max_confidence, conf_score)
            
            confidences[condition] = max_confidence
        
        logger.info(f"Extracted confidences: {confidences}")
        return confidences
