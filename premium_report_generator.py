"""
Premium Medical Report Generator - DeepMind Inspired
A visually stunning PDF report with rich colors, modern design, and professional aesthetics.
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
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as ReportLabImage, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


class PremiumReportGenerator:
    """
    DeepMind-inspired premium medical report generator.
    Rich visuals, color gradients, and professional medical aesthetics.
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
            self.font_title = 'Montserrat-Bold'
            self.font_body = 'Montserrat'
            self.font_bold = 'Montserrat-Bold'
            logger.info("Loaded premium fonts")
        except:
            self.font_title = 'Helvetica-Bold'
            self.font_body = 'Helvetica'
            self.font_bold = 'Helvetica-Bold'
    
    def _setup_colors(self):
        # DeepMind-inspired color palette
        self.colors = {
            # Primary - Deep blue gradient
            'primary_dark': colors.HexColor("#0D1B2A"),
            'primary': colors.HexColor("#1B263B"),
            'primary_light': colors.HexColor("#415A77"),
            
            # Accent - Vibrant cyan/teal
            'accent': colors.HexColor("#00D4FF"),
            'accent_dark': colors.HexColor("#00A0CC"),
            'accent_light': colors.HexColor("#80E9FF"),
            
            # Status colors
            'success': colors.HexColor("#00E676"),
            'success_dark': colors.HexColor("#00C853"),
            'success_bg': colors.HexColor("#E8F5E9"),
            
            'danger': colors.HexColor("#FF5252"),
            'danger_dark': colors.HexColor("#D32F2F"),
            'danger_bg': colors.HexColor("#FFEBEE"),
            
            'warning': colors.HexColor("#FFB300"),
            'warning_bg': colors.HexColor("#FFF8E1"),
            
            # Neutrals
            'dark': colors.HexColor("#121212"),
            'gray_900': colors.HexColor("#212121"),
            'gray_700': colors.HexColor("#424242"),
            'gray_500': colors.HexColor("#9E9E9E"),
            'gray_300': colors.HexColor("#E0E0E0"),
            'gray_100': colors.HexColor("#F5F5F5"),
            'white': colors.white,
            
            # Card backgrounds
            'card_bg': colors.HexColor("#FAFBFC"),
            'card_border': colors.HexColor("#E1E4E8"),
        }
    
    def _setup_styles(self):
        self.styles = {
            'hero_title': ParagraphStyle(
                'HeroTitle', fontName=self.font_title, fontSize=28,
                textColor=self.colors['white'], leading=34,
            ),
            'hero_subtitle': ParagraphStyle(
                'HeroSubtitle', fontName=self.font_body, fontSize=12,
                textColor=self.colors['accent_light'], leading=16,
            ),
            'section_title': ParagraphStyle(
                'SectionTitle', fontName=self.font_title, fontSize=14,
                textColor=self.colors['primary_dark'], spaceBefore=20, spaceAfter=12,
            ),
            'card_title': ParagraphStyle(
                'CardTitle', fontName=self.font_bold, fontSize=11,
                textColor=self.colors['gray_900'], spaceAfter=6,
            ),
            'body': ParagraphStyle(
                'Body', fontName=self.font_body, fontSize=10,
                textColor=self.colors['gray_700'], leading=14, spaceAfter=6,
            ),
            'body_small': ParagraphStyle(
                'BodySmall', fontName=self.font_body, fontSize=9,
                textColor=self.colors['gray_500'], leading=12,
            ),
            'status_detected': ParagraphStyle(
                'StatusDetected', fontName=self.font_bold, fontSize=11,
                textColor=self.colors['danger_dark'], alignment=TA_CENTER,
            ),
            'status_clear': ParagraphStyle(
                'StatusClear', fontName=self.font_bold, fontSize=11,
                textColor=self.colors['success_dark'], alignment=TA_CENTER,
            ),
            'caption': ParagraphStyle(
                'Caption', fontName=self.font_body, fontSize=9,
                textColor=self.colors['gray_500'], alignment=TA_CENTER, spaceBefore=4,
            ),
            'footer': ParagraphStyle(
                'Footer', fontName=self.font_body, fontSize=8,
                textColor=self.colors['gray_500'], leading=11,
            ),
        }
    
    def _pil_to_reportlab(self, pil_image: Image.Image, max_width: float, enhance: bool = False) -> ReportLabImage:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if enhance:
            # Boost contrast and saturation for heatmaps
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.4)
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(1.6)
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        aspect = pil_image.height / pil_image.width
        width = max_width
        height = min(width * aspect, 2.8 * inch)
        if height == 2.8 * inch:
            width = height / aspect
        
        return ReportLabImage(img_buffer, width=width, height=height)
    
    def _create_header_block(self, canvas, doc, analysis_id: str, timestamp: str):
        """Draw the dark hero header."""
        # Dark gradient background
        canvas.setFillColor(self.colors['primary_dark'])
        canvas.rect(0, doc.pagesize[1] - 120, doc.pagesize[0], 120, fill=1, stroke=0)
        
        # Accent line at bottom of header
        canvas.setStrokeColor(self.colors['accent'])
        canvas.setLineWidth(3)
        canvas.line(0, doc.pagesize[1] - 120, doc.pagesize[0], doc.pagesize[1] - 120)
        
        # Brand
        canvas.setFont(self.font_title, 16)
        canvas.setFillColor(self.colors['white'])
        canvas.drawString(40, doc.pagesize[1] - 40, "SENTRYAI")
        
        canvas.setFont(self.font_body, 9)
        canvas.setFillColor(self.colors['accent_light'])
        canvas.drawString(40, doc.pagesize[1] - 55, "MEDICAL INTELLIGENCE PLATFORM")
        
        # Title
        canvas.setFont(self.font_title, 22)
        canvas.setFillColor(self.colors['white'])
        canvas.drawString(40, doc.pagesize[1] - 90, "CHEST X-RAY ANALYSIS REPORT")
        
        # Metadata on right
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray_300'])
        canvas.drawRightString(doc.pagesize[0] - 40, doc.pagesize[1] - 40, f"ID: {analysis_id[:8]}...")
        canvas.drawRightString(doc.pagesize[0] - 40, doc.pagesize[1] - 52, timestamp)
    
    def _create_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(self.colors['gray_300'])
        canvas.setLineWidth(0.5)
        canvas.line(40, 40, doc.pagesize[0] - 40, 40)
        
        canvas.setFont(self.font_body, 7)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(40, 28, "CONFIDENTIAL • For licensed healthcare professionals only")
        canvas.drawRightString(doc.pagesize[0] - 40, 28, f"Page {doc.page}")
        canvas.restoreState()
    
    def _create_detection_card(self, condition: str, detected: bool, mg_agrees: bool, nv_agrees: bool) -> Table:
        """Create a rich detection card with color coding."""
        if detected:
            bg_color = self.colors['danger_bg']
            border_color = self.colors['danger']
            status_text = "DETECTED"
            icon = "⚠"
        else:
            bg_color = self.colors['success_bg']
            border_color = self.colors['success']
            status_text = "CLEAR"
            icon = "✓"
        
        # Agreement indicators
        mg_text = "✓ MedGemma" if mg_agrees else "✗ MedGemma"
        nv_text = "✓ NV-Reason" if nv_agrees else "✗ NV-Reason"
        mg_color = self.colors['success_dark'] if mg_agrees else self.colors['gray_500']
        nv_color = self.colors['success_dark'] if nv_agrees else self.colors['gray_500']
        
        card_content = [
            [Paragraph(f"<b>{condition}</b>", self.styles['card_title'])],
            [Paragraph(f"<font size='18'>{icon}</font> <b>{status_text}</b>",
                      ParagraphStyle('Status', parent=self.styles['body'],
                                    textColor=border_color, fontSize=14, alignment=TA_CENTER))],
            [Paragraph(f"<font color='{mg_color.hexval()}'>{mg_text}</font> | "
                      f"<font color='{nv_color.hexval()}'>{nv_text}</font>",
                      ParagraphStyle('Agreement', parent=self.styles['body_small'], alignment=TA_CENTER))],
        ]
        
        card = Table(card_content, colWidths=[2*inch])
        card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('LINEABOVE', (0, 0), (-1, 0), 3, border_color),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['card_border']),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return card
    
    def _create_consensus_badge(self, level: str) -> Table:
        """Create a visually striking consensus badge."""
        if level == 'high':
            bg_color = self.colors['success']
            text_color = self.colors['white']
            label = "HIGH AGREEMENT"
            desc = "Both AI models agree on findings"
        elif level == 'medium':
            bg_color = self.colors['warning']
            text_color = self.colors['dark']
            label = "PARTIAL AGREEMENT"
            desc = "Models have different findings - review recommended"
        else:
            bg_color = self.colors['danger']
            text_color = self.colors['white']
            label = "LOW AGREEMENT"
            desc = "Significant disagreement - clinical review required"
        
        badge_content = [
            [Paragraph(f"<b>{label}</b>",
                      ParagraphStyle('Badge', fontName=self.font_bold, fontSize=13,
                                    textColor=text_color, alignment=TA_CENTER))],
            [Paragraph(desc,
                      ParagraphStyle('Desc', fontName=self.font_body, fontSize=9,
                                    textColor=text_color, alignment=TA_CENTER))],
        ]
        
        badge = Table(badge_content, colWidths=[6*inch])
        badge.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('TOPPADDING', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return badge
    
    def generate_report(
        self,
        analysis_response,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        consensus_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate premium DeepMind-style PDF report."""
        
        filename = f"sentryai_medical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=130, bottomMargin=50
        )
        
        story = []
        findings = analysis_response.findings
        
        # Extract data
        if consensus_data:
            consensus_level = consensus_data.get('consensus_level', 'N/A')
            all_abnormalities = [a.lower() for a in consensus_data.get('all_abnormalities', [])]
            mg_abnormalities = [a.lower() for a in consensus_data.get('medgemma_abnormalities', [])]
            nv_abnormalities = [a.lower() for a in consensus_data.get('nv_reason_abnormalities', [])]
            clinical_impression = consensus_data.get('clinical_impression', '')
        else:
            consensus_level = 'single-model'
            all_abnormalities = [a.lower() for a in findings.abnormalities]
            mg_abnormalities = all_abnormalities
            nv_abnormalities = []
            clinical_impression = findings.impression
        
        # Store for header callback
        self._current_analysis_id = analysis_response.analysis_id
        self._current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # === IMAGING SECTION ===
        story.append(Paragraph("IMAGING ANALYSIS", self.styles['section_title']))
        
        img_width = 2.4 * inch
        rl_original = self._pil_to_reportlab(original_image, img_width)
        rl_heatmap = self._pil_to_reportlab(heatmap_image, img_width, enhance=True)
        
        # Images in styled container
        image_row = Table(
            [[rl_original, Spacer(0.4*inch, 0), rl_heatmap],
             [Paragraph("Original Radiograph", self.styles['caption']), 
              Spacer(0.4*inch, 0),
              Paragraph("AI Attention Map", self.styles['caption'])]],
            colWidths=[img_width + 0.1*inch, 0.4*inch, img_width + 0.1*inch]
        )
        image_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['card_bg']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['card_border']),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        story.append(image_row)
        story.append(Spacer(1, 20))
        
        # === DETECTION CARDS ===
        story.append(Paragraph("CONDITION SCREENING", self.styles['section_title']))
        
        conditions = [
            ('Pneumothorax', 'pneumothorax'),
            ('Pneumonia', 'pneumonia'),
            ('Tuberculosis', 'tb'),
        ]
        
        def is_real_detection(abnormalities_list, key):
            if not abnormalities_list:
                return False
            for a in abnormalities_list:
                a_lower = a.lower()
                if 'no evidence' in a_lower or 'not detected' in a_lower:
                    continue
                if key in a_lower:
                    return True
            return False
        
        # Create detection cards row
        cards = []
        for cond_name, cond_key in conditions:
            detected = is_real_detection(all_abnormalities, cond_key)
            mg = is_real_detection(mg_abnormalities, cond_key)
            nv = is_real_detection(nv_abnormalities, cond_key) if nv_abnormalities else False
            cards.append(self._create_detection_card(cond_name, detected, mg, nv))
        
        cards_row = Table([cards], colWidths=[2.15*inch, 2.15*inch, 2.15*inch])
        cards_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(cards_row)
        story.append(Spacer(1, 25))
        
        # === CONSENSUS SECTION ===
        story.append(Paragraph("MODEL CONSENSUS", self.styles['section_title']))
        story.append(self._create_consensus_badge(consensus_level))
        story.append(Spacer(1, 15))
        
        # Clinical Impression Card
        if clinical_impression:
            impression_card = Table(
                [[Paragraph("<b>Clinical Impression</b>", self.styles['card_title'])],
                 [Paragraph(clinical_impression, self.styles['body'])]],
                colWidths=[6.3*inch]
            )
            impression_card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['card_bg']),
                ('BOX', (0, 0), (-1, -1), 1, self.colors['card_border']),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ]))
            story.append(impression_card)
        
        story.append(Spacer(1, 30))
        
        # === DISCLAIMER ===
        story.append(HRFlowable(width="100%", thickness=1, color=self.colors['gray_300']))
        story.append(Spacer(1, 10))
        
        disclaimer = (
            "<b>Important Notice:</b> This AI-generated analysis is intended for clinical decision support only. "
            "All findings must be independently verified by a qualified radiologist or physician. "
            "AI models may have limitations and clinical judgment must supersede algorithmic output. "
            "This report does not constitute medical advice."
        )
        story.append(Paragraph(disclaimer, self.styles['footer']))
        
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | "
            f"<b>Models:</b> MedGemma-4B-IT + NV-Reason-CXR-3B",
            self.styles['footer']
        ))
        
        # Build with custom header
        doc.build(story, 
                  onFirstPage=lambda c, d: self._on_first_page(c, d),
                  onLaterPages=lambda c, d: self._on_later_pages(c, d))
        
        logger.info(f"Generated premium report: {filepath}")
        return filepath
    
    def _on_first_page(self, canvas, doc):
        self._create_header_block(canvas, doc, self._current_analysis_id, self._current_timestamp)
        self._create_footer(canvas, doc)
    
    def _on_later_pages(self, canvas, doc):
        # Simpler header for subsequent pages
        canvas.setFillColor(self.colors['primary_dark'])
        canvas.rect(0, doc.pagesize[1] - 50, doc.pagesize[0], 50, fill=1, stroke=0)
        canvas.setStrokeColor(self.colors['accent'])
        canvas.setLineWidth(2)
        canvas.line(0, doc.pagesize[1] - 50, doc.pagesize[0], doc.pagesize[1] - 50)
        
        canvas.setFont(self.font_title, 12)
        canvas.setFillColor(self.colors['white'])
        canvas.drawString(40, doc.pagesize[1] - 32, "SENTRYAI")
        
        canvas.setFont(self.font_body, 10)
        canvas.drawRightString(doc.pagesize[0] - 40, doc.pagesize[1] - 32, "Medical Analysis Report")
        
        self._create_footer(canvas, doc)


# Backwards compatibility
UltimatePDFGenerator = PremiumReportGenerator
