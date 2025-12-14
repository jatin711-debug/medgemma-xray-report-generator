"""
Simplified Medical Report Generator

Clean, detection-focused PDF report showing:
- Clear detection status for each condition
- Side-by-side model agreement
- Visible heatmap with original image
"""

import os
import io
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as ReportLabImage, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


class PremiumReportGenerator:
    """
    Clean, detection-focused medical report generator.
    Shows clear YES/NO for each condition with model agreement.
    """
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self._setup_fonts()
        self._setup_colors()
        self._setup_styles()
    
    def _setup_fonts(self):
        try:
            pdfmetrics.registerFont(TTFont('Montserrat', 'assets/Montserrat-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Montserrat-Bold', 'assets/Montserrat-Bold.ttf'))
            pdfmetrics.registerFont(TTFont('Source', 'assets/SourceSansPro-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Source-Bold', 'assets/SourceSansPro-Bold.ttf'))
            self.font_title = 'Montserrat-Bold'
            self.font_body = 'Source'
            self.font_bold = 'Source-Bold'
            logger.info("Loaded premium fonts")
        except:
            self.font_title = 'Helvetica-Bold'
            self.font_body = 'Helvetica'
            self.font_bold = 'Helvetica-Bold'
    
    def _setup_colors(self):
        self.colors = {
            'primary': colors.HexColor("#1A237E"),
            'primary_light': colors.HexColor("#3F51B5"),
            'success': colors.HexColor("#2E7D32"),
            'success_bg': colors.HexColor("#E8F5E9"),
            'danger': colors.HexColor("#C62828"),
            'danger_bg': colors.HexColor("#FFEBEE"),
            'warning': colors.HexColor("#F57C00"),
            'warning_bg': colors.HexColor("#FFF3E0"),
            'gray_dark': colors.HexColor("#424242"),
            'gray': colors.HexColor("#757575"),
            'gray_light': colors.HexColor("#E0E0E0"),
            'gray_bg': colors.HexColor("#FAFAFA"),
            'white': colors.white,
        }
    
    def _setup_styles(self):
        self.styles = {
            'title': ParagraphStyle('Title', fontName=self.font_title, fontSize=22,
                                   textColor=self.colors['primary'], spaceAfter=4),
            'subtitle': ParagraphStyle('Subtitle', fontName=self.font_body, fontSize=11,
                                       textColor=self.colors['gray'], spaceAfter=20),
            'section': ParagraphStyle('Section', fontName=self.font_title, fontSize=13,
                                      textColor=self.colors['primary'], spaceBefore=16, spaceAfter=10),
            'body': ParagraphStyle('Body', fontName=self.font_body, fontSize=10,
                                   textColor=self.colors['gray_dark'], leading=14, spaceAfter=6),
            'caption': ParagraphStyle('Caption', fontName=self.font_body, fontSize=9,
                                      textColor=self.colors['gray'], alignment=TA_CENTER, spaceBefore=4),
            'footer': ParagraphStyle('Footer', fontName=self.font_body, fontSize=8,
                                     textColor=self.colors['gray'], leading=11),
        }
    
    def _enhance_heatmap(self, heatmap: Image.Image) -> Image.Image:
        """Enhance heatmap visibility by boosting contrast and saturation."""
        if heatmap.mode != 'RGB':
            heatmap = heatmap.convert('RGB')
        
        # Boost contrast
        enhancer = ImageEnhance.Contrast(heatmap)
        heatmap = enhancer.enhance(1.3)
        
        # Boost saturation for more visible colors
        enhancer = ImageEnhance.Color(heatmap)
        heatmap = enhancer.enhance(1.5)
        
        return heatmap
    
    def _pil_to_reportlab(self, pil_image: Image.Image, max_width: float, enhance: bool = False) -> ReportLabImage:
        """Convert PIL to ReportLab image."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if enhance:
            pil_image = self._enhance_heatmap(pil_image)
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        aspect = pil_image.height / pil_image.width
        width = max_width
        height = min(width * aspect, 3 * inch)
        if height == 3 * inch:
            width = height / aspect
        
        return ReportLabImage(img_buffer, width=width, height=height)
    
    def _create_header(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(self.colors['primary'])
        canvas.setLineWidth(2)
        canvas.line(0, doc.pagesize[1] - 2, doc.pagesize[0], doc.pagesize[1] - 2)
        
        canvas.setFont(self.font_title, 11)
        canvas.setFillColor(self.colors['primary'])
        canvas.drawString(40, doc.pagesize[1] - 32, "SENTRYAI")
        
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray'])
        canvas.drawString(40, doc.pagesize[1] - 43, "Medical AI Analysis")
        
        canvas.setStrokeColor(self.colors['gray_light'])
        canvas.setLineWidth(0.5)
        canvas.line(40, doc.pagesize[1] - 50, doc.pagesize[0] - 40, doc.pagesize[1] - 50)
        canvas.restoreState()
    
    def _create_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(self.colors['gray_light'])
        canvas.setLineWidth(0.5)
        canvas.line(40, 45, doc.pagesize[0] - 40, 45)
        
        canvas.setFont(self.font_body, 7)
        canvas.setFillColor(self.colors['gray'])
        canvas.drawString(40, 32, "CONFIDENTIAL • For healthcare professionals only")
        canvas.drawRightString(doc.pagesize[0] - 40, 32, f"Page {doc.page}")
        canvas.restoreState()
    
    def _create_detection_row(self, condition: str, detected: bool, 
                              medgemma_agrees: bool, nv_reason_agrees: bool) -> List:
        """Create a detection status row."""
        # Status badge
        if detected:
            status_text = "DETECTED"
            status_color = self.colors['danger']
            status_bg = self.colors['danger_bg']
        else:
            status_text = "NOT DETECTED"
            status_color = self.colors['success']
            status_bg = self.colors['success_bg']
        
        # Agreement indicators
        mg_icon = "✓" if medgemma_agrees else "✗"
        nv_icon = "✓" if nv_reason_agrees else "✗"
        mg_color = self.colors['success'] if medgemma_agrees else self.colors['gray']
        nv_color = self.colors['success'] if nv_reason_agrees else self.colors['gray']
        
        return [
            Paragraph(f"<b>{condition}</b>", self.styles['body']),
            Paragraph(f"<font color='{status_color.hexval()}'><b>{status_text}</b></font>", 
                     ParagraphStyle('Status', parent=self.styles['body'], alignment=TA_CENTER)),
            Paragraph(f"<font color='{mg_color.hexval()}'>{mg_icon}</font>",
                     ParagraphStyle('Icon', parent=self.styles['body'], alignment=TA_CENTER, fontSize=12)),
            Paragraph(f"<font color='{nv_color.hexval()}'>{nv_icon}</font>",
                     ParagraphStyle('Icon', parent=self.styles['body'], alignment=TA_CENTER, fontSize=12)),
        ]
    
    def generate_report(
        self,
        analysis_response,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        consensus_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate clean detection-focused PDF report."""
        
        filename = f"sentryai_medical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=60, bottomMargin=55
        )
        
        story = []
        findings = analysis_response.findings
        
        # Extract data
        if consensus_data:
            consensus_level = consensus_data.get('consensus_level', 'N/A')
            all_abnormalities = [a.lower() for a in consensus_data.get('all_abnormalities', [])]
            mg_abnormalities = [a.lower() for a in consensus_data.get('medgemma_abnormalities', [])] if 'medgemma_abnormalities' in consensus_data else all_abnormalities
            nv_abnormalities = [a.lower() for a in consensus_data.get('nv_reason_abnormalities', [])] if 'nv_reason_abnormalities' in consensus_data else all_abnormalities
            clinical_impression = consensus_data.get('clinical_impression', '')
        else:
            consensus_level = 'single-model'
            all_abnormalities = [a.lower() for a in findings.abnormalities]
            mg_abnormalities = all_abnormalities
            nv_abnormalities = []
            clinical_impression = findings.impression
        
        # === HEADER ===
        story.append(Spacer(1, 10))
        story.append(Paragraph("CHEST X-RAY ANALYSIS REPORT", self.styles['title']))
        story.append(Paragraph(
            f"Analysis ID: {analysis_response.analysis_id} • "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles['subtitle']
        ))
        
        # === IMAGES SECTION ===
        story.append(Paragraph("IMAGING", self.styles['section']))
        
        img_width = 2.6 * inch
        rl_original = self._pil_to_reportlab(original_image, img_width)
        rl_heatmap = self._pil_to_reportlab(heatmap_image, img_width, enhance=True)
        
        image_table = Table(
            [[rl_original, Spacer(0.3*inch, 0), rl_heatmap],
             [Paragraph("Original X-Ray", self.styles['caption']), 
              Spacer(0.3*inch, 0),
              Paragraph("AI Attention Heatmap", self.styles['caption'])]],
            colWidths=[img_width + 0.1*inch, 0.3*inch, img_width + 0.1*inch]
        )
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 20))
        
        # === DETECTION RESULTS ===
        story.append(Paragraph("DETECTION RESULTS", self.styles['section']))
        story.append(Spacer(1, 6))
        
        # Only show these 3 conditions (restricted scope)
        conditions = [
            ('Pneumothorax', 'pneumothorax'),
            ('Pneumonia', 'pneumonia'),
            ('Tuberculosis (TB)', 'tb'),
        ]
        
        # Header row
        detection_data = [
            [Paragraph("<b>Condition</b>", self.styles['body']),
             Paragraph("<b>Status</b>", ParagraphStyle('H', parent=self.styles['body'], alignment=TA_CENTER)),
             Paragraph("<b>MedGemma</b>", ParagraphStyle('H', parent=self.styles['body'], alignment=TA_CENTER, fontSize=8)),
             Paragraph("<b>NV-Reason</b>", ParagraphStyle('H', parent=self.styles['body'], alignment=TA_CENTER, fontSize=8))],
        ]
        
        # Detection rows
        for display_name, key in conditions:
            detected = any(key in a for a in all_abnormalities)
            mg_agrees = any(key in a for a in mg_abnormalities)
            nv_agrees = any(key in a for a in nv_abnormalities) if nv_abnormalities else mg_agrees
            
            detection_data.append(self._create_detection_row(
                display_name, detected, mg_agrees, nv_agrees
            ))
        
        detection_table = Table(detection_data, colWidths=[2.5*inch, 1.8*inch, 1*inch, 1*inch])
        detection_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['gray_bg']),
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.colors['gray_light']),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, self.colors['gray_light']),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(detection_table)
        
        story.append(Spacer(1, 20))
        
        # === CONSENSUS ===
        story.append(Paragraph("MODEL CONSENSUS", self.styles['section']))
        
        if consensus_level == 'high':
            consensus_text = "Both AI models agree on the findings."
            consensus_color = self.colors['success']
            consensus_bg = self.colors['success_bg']
        elif consensus_level == 'medium':
            consensus_text = "Partial agreement between models. Review recommended."
            consensus_color = self.colors['warning']
            consensus_bg = self.colors['warning_bg']
        else:
            consensus_text = "Low agreement. Clinical review strongly recommended."
            consensus_color = self.colors['danger']
            consensus_bg = self.colors['danger_bg']
        
        consensus_box = Table(
            [[Paragraph(f"<b>Consensus Level: {consensus_level.upper()}</b><br/>{consensus_text}", 
                       ParagraphStyle('Consensus', parent=self.styles['body'], 
                                     textColor=consensus_color))]],
            colWidths=[6.3*inch]
        )
        consensus_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), consensus_bg),
            ('BOX', (0, 0), (-1, -1), 1, consensus_color),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ]))
        story.append(consensus_box)
        
        story.append(Spacer(1, 20))
        
        # === CLINICAL IMPRESSION ===
        if clinical_impression:
            story.append(Paragraph("CLINICAL IMPRESSION", self.styles['section']))
            story.append(Paragraph(clinical_impression, self.styles['body']))
        
        story.append(Spacer(1, 30))
        
        # === DISCLAIMER ===
        story.append(HRFlowable(width="100%", thickness=0.5, color=self.colors['gray_light']))
        story.append(Spacer(1, 10))
        
        disclaimer = (
            "<b>Important:</b> This AI analysis is for decision support only. "
            "All findings must be verified by a qualified radiologist. "
            "AI models may have limitations and clinical judgment must supersede algorithmic output."
        )
        story.append(Paragraph(disclaimer, self.styles['footer']))
        
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Generated by SentryAI Medical Intelligence Platform • "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            self.styles['footer']
        ))
        
        # Build
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)
        
        logger.info(f"Generated premium report: {filepath}")
        return filepath
    
    def _create_header_footer(self, canvas, doc):
        self._create_header(canvas, doc)
        self._create_footer(canvas, doc)


# Backwards compatibility
UltimatePDFGenerator = PremiumReportGenerator
