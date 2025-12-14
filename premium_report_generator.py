"""
Premium Medical Report Generator - DeepMind Style

A sophisticated PDF report generator for multi-model chest X-ray analysis
with proper pagination, full model outputs, and scientific aesthetic.
"""

import os
import io
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as ReportLabImage, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.widgets.markers import makeMarker
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class PremiumReportGenerator:
    """
    DeepMind-inspired premium medical report generator.
    
    Features:
    - Multi-page layout with proper text flow (no truncation)
    - Scientific color palette and typography
    - Side-by-side model comparison
    - Full AI reasoning output
    """
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self._setup_fonts()
        self._setup_colors()
        self._setup_styles()
    
    def _setup_fonts(self):
        """Load premium fonts or fallback to built-in."""
        try:
            pdfmetrics.registerFont(TTFont('Montserrat', 'assets/Montserrat-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Montserrat-Bold', 'assets/Montserrat-Bold.ttf'))
            pdfmetrics.registerFont(TTFont('Source', 'assets/SourceSansPro-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Source-Bold', 'assets/SourceSansPro-Bold.ttf'))
            self.font_display = 'Montserrat-Bold'
            self.font_heading = 'Montserrat'
            self.font_body = 'Source'
            self.font_body_bold = 'Source-Bold'
            logger.info("Loaded premium fonts")
        except Exception:
            logger.warning("Using fallback fonts")
            self.font_display = 'Helvetica-Bold'
            self.font_heading = 'Helvetica'
            self.font_body = 'Helvetica'
            self.font_body_bold = 'Helvetica-Bold'
    
    def _setup_colors(self):
        """DeepMind-inspired color palette."""
        self.colors = {
            # Primary blues
            'primary_dark': colors.HexColor("#1A237E"),
            'primary': colors.HexColor("#1976D2"),
            'primary_light': colors.HexColor("#64B5F6"),
            'primary_ultra_light': colors.HexColor("#E3F2FD"),
            
            # Scientific greens
            'success': colors.HexColor("#388E3C"),
            'success_light': colors.HexColor("#C8E6C9"),
            
            # Alerts
            'warning': colors.HexColor("#FB8C00"),
            'warning_light': colors.HexColor("#FFE0B2"),
            'danger': colors.HexColor("#E53935"),
            'danger_light': colors.HexColor("#FFCDD2"),
            
            # Grays
            'gray_900': colors.HexColor("#212121"),
            'gray_700': colors.HexColor("#616161"),
            'gray_500': colors.HexColor("#9E9E9E"),
            'gray_300': colors.HexColor("#E0E0E0"),
            'gray_100': colors.HexColor("#F5F5F5"),
            'gray_50': colors.HexColor("#FAFAFA"),
            
            'white': colors.white,
            'black': colors.black,
        }
    
    def _setup_styles(self):
        """Create paragraph styles."""
        self.styles = {
            'title': ParagraphStyle(
                'Title',
                fontName=self.font_display,
                fontSize=26,
                textColor=self.colors['primary_dark'],
                spaceAfter=6,
                leading=32,
            ),
            'subtitle': ParagraphStyle(
                'Subtitle',
                fontName=self.font_body,
                fontSize=12,
                textColor=self.colors['gray_500'],
                spaceAfter=24,
                leading=16,
            ),
            'section_header': ParagraphStyle(
                'SectionHeader',
                fontName=self.font_display,
                fontSize=14,
                textColor=self.colors['primary_dark'],
                spaceBefore=20,
                spaceAfter=12,
                leading=18,
            ),
            'subsection': ParagraphStyle(
                'Subsection',
                fontName=self.font_body_bold,
                fontSize=11,
                textColor=self.colors['gray_700'],
                spaceBefore=12,
                spaceAfter=6,
                leading=14,
            ),
            'body': ParagraphStyle(
                'Body',
                fontName=self.font_body,
                fontSize=10,
                textColor=self.colors['gray_700'],
                leading=16,
                spaceAfter=8,
            ),
            'body_small': ParagraphStyle(
                'BodySmall',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_500'],
                leading=14,
                spaceAfter=6,
            ),
            'model_output': ParagraphStyle(
                'ModelOutput',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_900'],
                leading=14,
                spaceAfter=4,
                leftIndent=10,
                rightIndent=10,
            ),
            'caption': ParagraphStyle(
                'Caption',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_500'],
                alignment=TA_CENTER,
                spaceBefore=6,
                spaceAfter=12,
            ),
            'disclaimer': ParagraphStyle(
                'Disclaimer',
                fontName=self.font_body,
                fontSize=8,
                textColor=self.colors['gray_500'],
                leading=12,
                spaceAfter=8,
            ),
        }
    
    def _create_header(self, canvas, doc):
        """Draw header on each page."""
        canvas.saveState()
        
        # Top accent line
        canvas.setStrokeColor(self.colors['primary'])
        canvas.setLineWidth(2)
        canvas.line(0, doc.pagesize[1] - 2, doc.pagesize[0], doc.pagesize[1] - 2)
        
        # Logo area
        canvas.setFont(self.font_display, 12)
        canvas.setFillColor(self.colors['primary_dark'])
        canvas.drawString(40, doc.pagesize[1] - 35, "SENTRYAI")
        
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(40, doc.pagesize[1] - 47, "Medical Intelligence Platform")
        
        # Separator
        canvas.setStrokeColor(self.colors['gray_300'])
        canvas.setLineWidth(0.5)
        canvas.line(40, doc.pagesize[1] - 55, doc.pagesize[0] - 40, doc.pagesize[1] - 55)
        
        canvas.restoreState()
    
    def _create_footer(self, canvas, doc):
        """Draw footer on each page."""
        canvas.saveState()
        
        # Separator
        canvas.setStrokeColor(self.colors['gray_300'])
        canvas.setLineWidth(0.5)
        canvas.line(40, 50, doc.pagesize[0] - 40, 50)
        
        # Footer text
        canvas.setFont(self.font_body, 7)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(40, 35, "CONFIDENTIAL MEDICAL ANALYSIS • For licensed healthcare professionals only")
        
        # Page number
        canvas.setFont(self.font_body_bold, 9)
        canvas.setFillColor(self.colors['primary'])
        canvas.drawRightString(doc.pagesize[0] - 40, 35, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _draw_page_template(self, canvas, doc):
        """Combined header and footer."""
        self._create_header(canvas, doc)
        self._create_footer(canvas, doc)
    
    def _pil_to_reportlab(self, pil_image: Image.Image, max_width: float) -> ReportLabImage:
        """Convert PIL image to ReportLab with proper sizing."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Light sharpening
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=80))
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        aspect = pil_image.height / pil_image.width
        width = max_width
        height = width * aspect
        
        # Limit height
        max_height = 3.5 * inch
        if height > max_height:
            height = max_height
            width = height / aspect
        
        return ReportLabImage(img_buffer, width=width, height=height)
    
    def _create_status_card(self, status: str, impression: str, confidence: str) -> Table:
        """Create executive summary status card."""
        is_positive = any(w in status.lower() for w in ['abnormal', 'detected', 'positive'])
        
        if is_positive:
            bg_color = self.colors['danger_light']
            border_color = self.colors['danger']
            icon = "⚠"
            status_text = "ABNORMALITIES DETECTED"
        else:
            bg_color = self.colors['success_light']
            border_color = self.colors['success']
            icon = "✓"
            status_text = "NO SIGNIFICANT ABNORMALITIES"
        
        content = [
            [Paragraph(f"<b>{icon} {status_text}</b>", 
                      ParagraphStyle('Status', parent=self.styles['subsection'], 
                                    textColor=border_color, fontSize=12))],
            [Paragraph(f"<b>Clinical Impression:</b> {impression}", self.styles['body'])],
            [Paragraph(f"<b>Confidence Level:</b> {confidence}", self.styles['body'])],
        ]
        
        table = Table(content, colWidths=[6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('LINEABOVE', (0, 0), (-1, 0), 3, border_color),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_300']),
            ('TOPPADDING', (0, 0), (-1, -1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ]))
        
        return table
    
    def _create_metric_cards(self, consensus_level: str, models_count: int, conditions_count: int) -> Table:
        """Create row of metric cards."""
        def make_card(value, label):
            return Table(
                [[Paragraph(f"<b>{value}</b>", 
                           ParagraphStyle('MetricValue', parent=self.styles['body'],
                                         fontSize=16, textColor=self.colors['primary_dark'],
                                         alignment=TA_CENTER))],
                 [Paragraph(label, 
                           ParagraphStyle('MetricLabel', parent=self.styles['body_small'],
                                         alignment=TA_CENTER))]],
                colWidths=[2*inch]
            )
        
        cards = [
            make_card(consensus_level.upper(), "Consensus Level"),
            make_card(str(models_count), "Models Used"),
            make_card(str(conditions_count), "Conditions Screened"),
        ]
        
        row = Table([cards], colWidths=[2.1*inch, 2.1*inch, 2.1*inch])
        row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return row
    
    def _create_model_comparison(self, medgemma_summary: str, nv_reason_summary: str) -> Table:
        """Create side-by-side model comparison cards."""
        # Truncate for preview only (full text on later pages)
        mg_preview = medgemma_summary[:300] + "..." if len(medgemma_summary) > 300 else medgemma_summary
        nv_preview = nv_reason_summary[:300] + "..." if len(nv_reason_summary) > 300 else nv_reason_summary
        
        mg_card = [
            [Paragraph("<b>MedGemma-4B-IT</b>", self.styles['subsection'])],
            [Paragraph(mg_preview.replace('\n', '<br/>'), self.styles['body_small'])],
        ]
        
        nv_card = [
            [Paragraph("<b>NV-Reason-CXR-3B</b>", self.styles['subsection'])],
            [Paragraph(nv_preview.replace('\n', '<br/>'), self.styles['body_small'])],
        ]
        
        mg_table = Table(mg_card, colWidths=[3*inch])
        nv_table = Table(nv_card, colWidths=[3*inch])
        
        for t in [mg_table, nv_table]:
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
                ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_300']),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('LEFTPADDING', (0, 0), (-1, -1), 14),
                ('RIGHTPADDING', (0, 0), (-1, -1), 14),
            ]))
        
        comparison = Table([[mg_table, Spacer(0.2*inch, 0), nv_table]], 
                          colWidths=[3.1*inch, 0.2*inch, 3.1*inch])
        return comparison
    
    def generate_report(
        self,
        analysis_response,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        consensus_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate premium multi-page PDF report.
        
        Args:
            analysis_response: Analysis response with findings
            original_image: Original X-ray image
            heatmap_image: AI attention heatmap
            consensus_data: Multi-model consensus data (optional)
        
        Returns:
            Path to generated PDF file
        """
        filename = f"sentryai_medical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=70, bottomMargin=60
        )
        
        story = []
        findings = analysis_response.findings
        
        # Extract consensus data if available
        if consensus_data:
            medgemma_raw = consensus_data.get('medgemma_raw', '')
            nv_reason_thinking = consensus_data.get('nv_reason_thinking', '')
            nv_reason_answer = consensus_data.get('nv_reason_answer', '')
            consensus_level = consensus_data.get('consensus_level', 'N/A')
            clinical_impression = consensus_data.get('clinical_impression', findings.impression)
            all_abnormalities = consensus_data.get('all_abnormalities', [])
            nv_devices = consensus_data.get('nv_reason_devices', [])
        else:
            medgemma_raw = analysis_response.raw_response
            nv_reason_thinking = ''
            nv_reason_answer = ''
            consensus_level = 'single-model'
            clinical_impression = findings.impression
            all_abnormalities = findings.abnormalities
            nv_devices = []
        
        # ============ PAGE 1: COVER ============
        story.append(Spacer(1, 30))
        story.append(Paragraph("CHEST RADIOGRAPHY ANALYSIS", self.styles['title']))
        story.append(Paragraph("Multi-Model AI Diagnostic Report", self.styles['subtitle']))
        
        story.append(Spacer(1, 20))
        
        # Executive Summary Card
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['section_header']))
        story.append(Spacer(1, 10))
        story.append(self._create_status_card(
            clinical_impression,
            clinical_impression.replace('_', ' ').title(),
            findings.confidence_level
        ))
        
        story.append(Spacer(1, 30))
        
        # Metric Cards
        story.append(self._create_metric_cards(
            consensus_level,
            2 if consensus_data else 1,
            3
        ))
        
        story.append(Spacer(1, 40))
        
        # Study Info
        story.append(Paragraph("STUDY INFORMATION", self.styles['section_header']))
        story.append(Spacer(1, 8))
        
        meta = analysis_response.image_metadata or {}
        study_info = [
            ('Analysis ID', analysis_response.analysis_id),
            ('Timestamp', analysis_response.timestamp),
            ('Models', 'MedGemma-4B-IT + NV-Reason-CXR-3B' if consensus_data else 'MedGemma-4B-IT'),
        ]
        
        for label, value in study_info:
            story.append(Paragraph(f"<b>{label}:</b> {value}", self.styles['body_small']))
        
        story.append(PageBreak())
        
        # ============ PAGE 2: IMAGING ============
        story.append(Paragraph("IMAGING ANALYSIS", self.styles['section_header']))
        story.append(Spacer(1, 16))
        
        # Side-by-side images
        img_width = 2.8 * inch
        rl_original = self._pil_to_reportlab(original_image, img_width)
        rl_heatmap = self._pil_to_reportlab(heatmap_image, img_width)
        
        image_row = Table(
            [[rl_original, Spacer(0.2*inch, 0), rl_heatmap],
             [Paragraph("Original X-Ray", self.styles['caption']), 
              Spacer(0.2*inch, 0),
              Paragraph("AI Attention Heatmap", self.styles['caption'])]],
            colWidths=[img_width + 0.1*inch, 0.2*inch, img_width + 0.1*inch]
        )
        image_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
        ]))
        story.append(image_row)
        
        story.append(Spacer(1, 16))
        story.append(Paragraph(
            "<i>The attention heatmap visualizes regions of highest diagnostic relevance. "
            "Warmer colors (red/yellow) indicate areas of greater model focus.</i>",
            self.styles['body_small']
        ))
        
        story.append(Spacer(1, 30))
        
        # Model Comparison Preview (if multi-model)
        if consensus_data:
            story.append(Paragraph("MODEL COMPARISON", self.styles['section_header']))
            story.append(Spacer(1, 12))
            story.append(self._create_model_comparison(
                medgemma_raw or "Analysis not available",
                nv_reason_answer or "Analysis not available"
            ))
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                "<i>See following pages for complete model outputs.</i>",
                self.styles['body_small']
            ))
        
        story.append(PageBreak())
        
        # ============ PAGE 3+: FULL ANALYSIS ============
        story.append(Paragraph("DETAILED AI ANALYSIS", self.styles['section_header']))
        story.append(Spacer(1, 16))
        
        # NV-Reason Full Output (if available)
        if nv_reason_thinking or nv_reason_answer:
            story.append(Paragraph("NV-Reason-CXR-3B Clinical Reasoning", self.styles['subsection']))
            story.append(Spacer(1, 8))
            
            if nv_reason_thinking:
                story.append(Paragraph("<b>Thinking Chain:</b>", self.styles['body']))
                # Split into paragraphs for proper flow
                for para in nv_reason_thinking.split('\n\n'):
                    if para.strip():
                        story.append(Paragraph(para.replace('\n', ' '), self.styles['model_output']))
                story.append(Spacer(1, 12))
            
            if nv_reason_answer:
                story.append(Paragraph("<b>Final Assessment:</b>", self.styles['body']))
                for para in nv_reason_answer.split('\n\n'):
                    if para.strip():
                        story.append(Paragraph(para.replace('\n', ' '), self.styles['model_output']))
            
            story.append(Spacer(1, 24))
        
        # MedGemma Full Output
        story.append(Paragraph("MedGemma-4B-IT Analysis", self.styles['subsection']))
        story.append(Spacer(1, 8))
        
        if medgemma_raw:
            for para in medgemma_raw.split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para.replace('\n', ' '), self.styles['model_output']))
        else:
            story.append(Paragraph("Analysis not available", self.styles['body_small']))
        
        story.append(Spacer(1, 24))
        
        # Findings Summary
        story.append(Paragraph("FINDINGS SUMMARY", self.styles['section_header']))
        story.append(Spacer(1, 12))
        
        conditions = ['Pneumothorax', 'Pneumonia', 'TB']
        for condition in conditions:
            detected = any(condition.lower() in a.lower() for a in all_abnormalities)
            status = "DETECTED" if detected else "Not Detected"
            status_color = self.colors['danger'] if detected else self.colors['success']
            
            story.append(Paragraph(
                f"<b>{condition}:</b> <font color='{status_color.hexval()}'>{status}</font>",
                self.styles['body']
            ))
        
        if nv_devices:
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"<b>Support Devices:</b> {', '.join(nv_devices)}",
                self.styles['body']
            ))
        
        story.append(PageBreak())
        
        # ============ FINAL PAGE: RECOMMENDATIONS ============
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['section_header']))
        story.append(Spacer(1, 16))
        
        recommendations = [
            ("<b>Physician Review Required:</b> All AI findings must be independently "
             "verified by a qualified radiologist before clinical action."),
            ("<b>Clinical Correlation:</b> Imaging findings should be interpreted "
             "alongside patient history and physical examination."),
            ("<b>Model Limitations:</b> AI models may miss subtle findings or rare "
             "conditions. Clinical judgment must supersede algorithmic output."),
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['body']))
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 30))
        
        # Attestation
        story.append(HRFlowable(width="100%", thickness=0.5, color=self.colors['gray_300']))
        story.append(Spacer(1, 12))
        
        attestation = (
            f"This report was automatically generated by the SentryAI Medical Intelligence Platform "
            f"using deep learning models trained on medical imaging datasets. "
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        story.append(Paragraph(attestation, self.styles['disclaimer']))
        
        # Build PDF
        doc.build(story, onFirstPage=self._draw_page_template, onLaterPages=self._draw_page_template)
        
        logger.info(f"Generated premium report: {filepath}")
        return filepath


# Backwards compatible alias
UltimatePDFGenerator = PremiumReportGenerator
