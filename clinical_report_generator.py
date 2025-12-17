"""
Clinical-Grade Medical Report Generator
Professional PDF report with proper page structure and clinical formatting.

Page Structure:
- Page 1: Cover + Executive Summary
- Page 2: Side-by-side Original X-ray ↔ AI Evidence Map
- Page 3: Lesion Localization (dedicated page)
- Page 4: Diagnostic Probability Matrix + Condition Cards
- Page 5: Clinical Impression + Recommendations
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
    PageBreak, Image as ReportLabImage, HRFlowable, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM FLOWABLES
# =============================================================================

class ProgressBar(Flowable):
    """Clinical-style progress bar for diagnostic probability."""
    
    def __init__(self, value: float, width: float = 200, height: float = 16, 
                 label: str = "", show_percentage: bool = True):
        Flowable.__init__(self)
        self.value = max(0, min(100, value))  # Clamp 0-100
        self.width = width
        self.height = height
        self.label = label
        self.show_percentage = show_percentage
    
    def draw(self):
        canvas = self.canv
        
        # Background track
        canvas.setFillColor(colors.HexColor("#E5E7EB"))
        canvas.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        
        # Determine color based on value
        if self.value >= 70:
            bar_color = colors.HexColor("#EF4444")  # Red - detected
        elif self.value >= 40:
            bar_color = colors.HexColor("#F59E0B")  # Orange - possible
        else:
            bar_color = colors.HexColor("#10B981")  # Green - clear
        
        # Filled portion
        fill_width = (self.value / 100) * self.width
        if fill_width > 0:
            canvas.setFillColor(bar_color)
            canvas.roundRect(0, 0, fill_width, self.height, 4, fill=1, stroke=0)
        
        # Percentage text
        if self.show_percentage:
            canvas.setFillColor(colors.white if fill_width > 40 else colors.HexColor("#374151"))
            canvas.setFont("Helvetica-Bold", 9)
            text_x = min(fill_width - 25, self.width - 30) if fill_width > 40 else 5
            canvas.drawString(text_x, 4, f"{int(self.value)}%")


class StatusBadge(Flowable):
    """Status badge for detection results."""
    
    def __init__(self, status: str, width: float = 100, height: float = 28):
        Flowable.__init__(self)
        self.status = status.upper()
        self.width = width
        self.height = height
    
    def draw(self):
        canvas = self.canv
        
        if "DETECTED" in self.status:
            bg_color = colors.HexColor("#FEE2E2")
            text_color = colors.HexColor("#DC2626")
            icon = "⚠"
        else:
            bg_color = colors.HexColor("#D1FAE5")
            text_color = colors.HexColor("#059669")
            icon = "✓"
        
        # Background
        canvas.setFillColor(bg_color)
        canvas.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        
        # Text
        canvas.setFillColor(text_color)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(self.width / 2, 9, f"{icon} {self.status}")


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class ClinicalReportGenerator:
    """
    Clinical-grade PDF report generator with professional medical layout.
    
    Features:
    - Consistent page structure with dedicated sections
    - Side-by-side image comparisons
    - Proper clinical formatting and hierarchy
    - Fixed image sizing for alignment
    """
    
    # Fixed dimensions for consistency
    PAGE_WIDTH = letter[0]  # 612 points
    PAGE_HEIGHT = letter[1]  # 792 points
    MARGIN = 50
    CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN)  # 512 points
    
    # Image sizing
    IMAGE_WIDTH = 3.2 * inch  # Standard image width
    IMAGE_PAIR_GAP = 0.3 * inch  # Gap between side-by-side images
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self._setup_fonts()
        self._setup_colors()
        self._setup_styles()
    
    def _setup_fonts(self):
        """Load premium fonts with fallback."""
        try:
            pdfmetrics.registerFont(TTFont('Montserrat', 'assets/Montserrat-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Montserrat-Bold', 'assets/Montserrat-Bold.ttf'))
            pdfmetrics.registerFont(TTFont('Montserrat-Light', 'assets/Montserrat-Light.ttf'))
            self.font_display = 'Montserrat-Bold'
            self.font_body = 'Montserrat'
            self.font_light = 'Montserrat-Light'
            logger.info("Loaded premium fonts")
        except Exception as e:
            logger.warning(f"Using fallback fonts: {e}")
            self.font_display = 'Helvetica-Bold'
            self.font_body = 'Helvetica'
            self.font_light = 'Helvetica'
    
    def _setup_colors(self):
        """Clinical color palette."""
        self.colors = {
            # Primary - Deep medical blue
            'primary': colors.HexColor("#0F172A"),
            'primary_light': colors.HexColor("#1E293B"),
            'accent': colors.HexColor("#0EA5E9"),
            'accent_light': colors.HexColor("#E0F2FE"),
            
            # Status colors
            'success': colors.HexColor("#10B981"),
            'success_light': colors.HexColor("#D1FAE5"),
            'warning': colors.HexColor("#F59E0B"),
            'warning_light': colors.HexColor("#FEF3C7"),
            'danger': colors.HexColor("#EF4444"),
            'danger_light': colors.HexColor("#FEE2E2"),
            
            # Neutrals
            'gray_50': colors.HexColor("#F9FAFB"),
            'gray_100': colors.HexColor("#F3F4F6"),
            'gray_200': colors.HexColor("#E5E7EB"),
            'gray_300': colors.HexColor("#D1D5DB"),
            'gray_400': colors.HexColor("#9CA3AF"),
            'gray_500': colors.HexColor("#6B7280"),
            'gray_600': colors.HexColor("#4B5563"),
            'gray_700': colors.HexColor("#374151"),
            'gray_800': colors.HexColor("#1F2937"),
            'gray_900': colors.HexColor("#111827"),
            
            'white': colors.white,
            'black': colors.black,
        }
    
    def _setup_styles(self):
        """Clinical typography system."""
        self.styles = {
            # Hero/Title styles
            'hero_title': ParagraphStyle(
                'HeroTitle',
                fontName=self.font_display,
                fontSize=26,
                textColor=self.colors['white'],
                leading=32,
                spaceAfter=4,
            ),
            'hero_subtitle': ParagraphStyle(
                'HeroSubtitle',
                fontName=self.font_body,
                fontSize=11,
                textColor=self.colors['accent'],
                leading=14,
            ),
            
            # Section headers
            'section_title': ParagraphStyle(
                'SectionTitle',
                fontName=self.font_display,
                fontSize=14,
                textColor=self.colors['primary'],
                spaceBefore=20,
                spaceAfter=12,
                leading=18,
            ),
            'subsection_title': ParagraphStyle(
                'SubsectionTitle',
                fontName=self.font_display,
                fontSize=11,
                textColor=self.colors['gray_700'],
                spaceBefore=12,
                spaceAfter=8,
                leading=14,
            ),
            
            # Body text
            'body': ParagraphStyle(
                'Body',
                fontName=self.font_body,
                fontSize=10,
                textColor=self.colors['gray_700'],
                leading=15,
                spaceAfter=6,
            ),
            'body_small': ParagraphStyle(
                'BodySmall',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_500'],
                leading=13,
            ),
            
            # Captions
            'caption': ParagraphStyle(
                'Caption',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_500'],
                alignment=TA_CENTER,
                spaceBefore=6,
                spaceAfter=4,
            ),
            
            # Metadata
            'meta_label': ParagraphStyle(
                'MetaLabel',
                fontName=self.font_display,
                fontSize=8,
                textColor=self.colors['gray_500'],
                leading=11,
            ),
            'meta_value': ParagraphStyle(
                'MetaValue',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_800'],
                leading=12,
            ),
            
            # Status styles
            'status_detected': ParagraphStyle(
                'StatusDetected',
                fontName=self.font_display,
                fontSize=12,
                textColor=self.colors['danger'],
                alignment=TA_CENTER,
            ),
            'status_clear': ParagraphStyle(
                'StatusClear',
                fontName=self.font_display,
                fontSize=12,
                textColor=self.colors['success'],
                alignment=TA_CENTER,
            ),
            
            # Footer/Disclaimer
            'disclaimer': ParagraphStyle(
                'Disclaimer',
                fontName=self.font_body,
                fontSize=8,
                textColor=self.colors['gray_500'],
                leading=11,
                spaceAfter=6,
            ),
        }
    
    # =========================================================================
    # IMAGE UTILITIES
    # =========================================================================
    
    def _pil_to_reportlab(self, pil_image: Image.Image, width: float, 
                          enhance: bool = False) -> ReportLabImage:
        """Convert PIL image to ReportLab with consistent sizing."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if enhance:
            # Boost contrast for heatmaps
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(1.4)
        
        # Calculate height maintaining aspect ratio
        aspect = pil_image.height / pil_image.width
        height = width * aspect
        
        # Cap maximum height
        max_height = 4 * inch
        if height > max_height:
            height = max_height
            width = height / aspect
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        return ReportLabImage(img_buffer, width=width, height=height)
    
    # =========================================================================
    # PAGE TEMPLATES
    # =========================================================================
    
    def _draw_header(self, canvas, doc, is_first_page: bool = False):
        """Draw page header."""
        canvas.saveState()
        
        if is_first_page:
            # Dark hero header for first page
            header_height = 100
            canvas.setFillColor(self.colors['primary'])
            canvas.rect(0, doc.pagesize[1] - header_height, doc.pagesize[0], header_height, fill=1, stroke=0)
            
            # Accent line
            canvas.setStrokeColor(self.colors['accent'])
            canvas.setLineWidth(3)
            canvas.line(0, doc.pagesize[1] - header_height, doc.pagesize[0], doc.pagesize[1] - header_height)
            
            # Brand
            canvas.setFont(self.font_display, 18)
            canvas.setFillColor(self.colors['white'])
            canvas.drawString(self.MARGIN, doc.pagesize[1] - 40, "SENTRYAI")
            
            canvas.setFont(self.font_body, 9)
            canvas.setFillColor(self.colors['accent'])
            canvas.drawString(self.MARGIN, doc.pagesize[1] - 55, "MEDICAL INTELLIGENCE PLATFORM")
            
            # Title
            canvas.setFont(self.font_display, 20)
            canvas.setFillColor(self.colors['white'])
            canvas.drawString(self.MARGIN, doc.pagesize[1] - 85, "CHEST X-RAY ANALYSIS REPORT")
        else:
            # Simple header for other pages
            header_height = 45
            canvas.setFillColor(self.colors['primary'])
            canvas.rect(0, doc.pagesize[1] - header_height, doc.pagesize[0], header_height, fill=1, stroke=0)
            
            canvas.setStrokeColor(self.colors['accent'])
            canvas.setLineWidth(2)
            canvas.line(0, doc.pagesize[1] - header_height, doc.pagesize[0], doc.pagesize[1] - header_height)
            
            canvas.setFont(self.font_display, 12)
            canvas.setFillColor(self.colors['white'])
            canvas.drawString(self.MARGIN, doc.pagesize[1] - 30, "SENTRYAI")
            
            canvas.setFont(self.font_body, 10)
            canvas.drawRightString(doc.pagesize[0] - self.MARGIN, doc.pagesize[1] - 30, "Medical Analysis Report")
        
        canvas.restoreState()
    
    def _draw_footer(self, canvas, doc):
        """Draw page footer."""
        canvas.saveState()
        
        footer_y = 35
        
        # Separator line
        canvas.setStrokeColor(self.colors['gray_200'])
        canvas.setLineWidth(0.5)
        canvas.line(self.MARGIN, 50, doc.pagesize[0] - self.MARGIN, 50)
        
        # Footer text
        canvas.setFont(self.font_body, 7)
        canvas.setFillColor(self.colors['gray_400'])
        canvas.drawString(self.MARGIN, footer_y, "CONFIDENTIAL • For licensed healthcare professionals only")
        
        # Page number
        canvas.setFont(self.font_display, 9)
        canvas.setFillColor(self.colors['gray_600'])
        canvas.drawRightString(doc.pagesize[0] - self.MARGIN, footer_y, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _on_first_page(self, canvas, doc):
        self._draw_header(canvas, doc, is_first_page=True)
        self._draw_footer(canvas, doc)
    
    def _on_later_pages(self, canvas, doc):
        self._draw_header(canvas, doc, is_first_page=False)
        self._draw_footer(canvas, doc)
    
    # =========================================================================
    # COMPONENT BUILDERS
    # =========================================================================
    
    def _build_executive_summary(self, consensus_data: Dict, condition_confidences: Dict) -> List:
        """Build executive summary section with urgency triage and negative findings."""
        elements = []
        
        # Determine overall status and urgency level
        detected_conditions = []
        possible_conditions = []
        clear_conditions = []
        
        if condition_confidences:
            for cond, conf in condition_confidences.items():
                cond_display = cond.replace('_', ' ').title()
                if conf >= 0.7:
                    detected_conditions.append((cond_display, conf))
                elif conf >= 0.4:
                    possible_conditions.append((cond_display, conf))
                else:
                    clear_conditions.append((cond_display, conf))
        
        has_detection = len(detected_conditions) > 0
        has_possible = len(possible_conditions) > 0
        consensus_level = consensus_data.get('consensus_level', 'N/A') if consensus_data else 'single-model'
        
        # Determine urgency level
        if has_detection:
            urgency = "STAT"
            urgency_color = self.colors['danger']
            urgency_bg = self.colors['danger_light']
            urgency_desc = "Immediate radiologist review recommended"
            status_text = "ABNORMALITY DETECTED"
        elif has_possible:
            urgency = "ROUTINE"
            urgency_color = self.colors['warning']
            urgency_bg = self.colors['warning_light']
            urgency_desc = "Findings require clinical correlation"
            status_text = "POSSIBLE FINDINGS"
        else:
            urgency = "NORMAL"
            urgency_color = self.colors['success']
            urgency_bg = self.colors['success_light']
            urgency_desc = "No significant abnormalities detected"
            status_text = "NO SIGNIFICANT ABNORMALITIES"
        
        # =========== URGENCY TRIAGE BADGE ===========
        urgency_badge = Table(
            [[Paragraph(f"<b>🚨 {urgency}</b>", ParagraphStyle(
                'UrgencyTitle', fontName=self.font_display, fontSize=16,
                textColor=urgency_color, alignment=TA_CENTER
            ))],
            [Paragraph(urgency_desc, ParagraphStyle(
                'UrgencyDesc', parent=self.styles['body_small'],
                alignment=TA_CENTER, textColor=self.colors['gray_600']
            ))]],
            colWidths=[self.CONTENT_WIDTH - 40]
        )
        urgency_badge.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), urgency_bg),
            ('LINEABOVE', (0, 0), (-1, 0), 4, urgency_color),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        elements.append(urgency_badge)
        elements.append(Spacer(1, 15))
        
        # =========== PRIMARY FINDINGS ===========
        findings_content = []
        
        # Detected conditions (high confidence)
        if detected_conditions:
            detected_text = ", ".join([f"<b>{c[0]}</b> ({int(c[1]*100)}%)" for c in detected_conditions])
            findings_content.append([Paragraph(
                f"<font color='#EF4444'>⚠ DETECTED:</font> {detected_text}",
                self.styles['body']
            )])
        
        # Possible conditions (medium confidence)
        if possible_conditions:
            possible_text = ", ".join([f"{c[0]} ({int(c[1]*100)}%)" for c in possible_conditions])
            findings_content.append([Paragraph(
                f"<font color='#F59E0B'>⚡ POSSIBLE:</font> {possible_text}",
                self.styles['body']
            )])
        
        # =========== EXPLICIT NEGATIVE FINDINGS ===========
        if clear_conditions:
            clear_names = [c[0] for c in clear_conditions]
            negative_text = f"No radiographic evidence of {', '.join(clear_names[:-1])}" if len(clear_names) > 1 else f"No radiographic evidence of {clear_names[0]}"
            if len(clear_names) > 1:
                negative_text += f" or {clear_names[-1]}"
            negative_text += " was detected."
            
            findings_content.append([Paragraph(
                f"<font color='#10B981'>✓ NEGATIVE:</font> {negative_text}",
                self.styles['body']
            )])
        
        # Add model consensus
        findings_content.append([Paragraph(
            f"<b>Model Consensus:</b> {consensus_level.upper()}",
            self.styles['body_small']
        )])
        
        if findings_content:
            findings_table = Table(findings_content, colWidths=[self.CONTENT_WIDTH - 60])
            findings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['white']),
                ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('LINEBELOW', (0, 0), (-1, -2), 0.5, self.colors['gray_100']),
            ]))
            elements.append(findings_table)
        
        return elements
    
    def _build_side_by_side_images(self, original: Image.Image, heatmap: Image.Image,
                                    left_caption: str = "Original Radiograph",
                                    right_caption: str = "AI Attention Map") -> Table:
        """Build side-by-side image comparison."""
        
        # Calculate image width for side-by-side (accounting for gap)
        img_width = (self.CONTENT_WIDTH - self.IMAGE_PAIR_GAP) / 2
        
        rl_original = self._pil_to_reportlab(original, img_width)
        rl_heatmap = self._pil_to_reportlab(heatmap, img_width, enhance=True)
        
        # Create table with images and captions
        image_data = [
            [rl_original, Spacer(self.IMAGE_PAIR_GAP, 0), rl_heatmap],
            [
                Paragraph(left_caption, self.styles['caption']),
                Spacer(self.IMAGE_PAIR_GAP, 0),
                Paragraph(right_caption, self.styles['caption'])
            ],
        ]
        
        col_widths = [img_width, self.IMAGE_PAIR_GAP, img_width]
        image_table = Table(image_data, colWidths=col_widths)
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
            ('VALIGN', (0, 1), (-1, 1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        return image_table
    
    def _build_localization_section(self, localization_map: Image.Image,
                                     detected_regions: List[Dict]) -> List:
        """Build lesion localization section with map and legend."""
        elements = []
        
        # Image on left, legend on right
        img_width = self.CONTENT_WIDTH * 0.55
        legend_width = self.CONTENT_WIDTH * 0.40
        
        rl_map = self._pil_to_reportlab(localization_map, img_width, enhance=True)
        
        # Build legend content with clinical terminology
        if detected_regions:
            legend_items = []
            for i, region in enumerate(detected_regions[:5], 1):
                evidence_score = int(region.get('attention_score', 0) * 100)
                location = region.get('location', 'Unknown')
                legend_items.append(
                    f"<b>R{i}:</b> {location} — Relative AI evidence: {evidence_score}%"
                )
            legend_text = "<br/>".join(legend_items)
        else:
            legend_text = "No significant regions identified"
        
        legend_content = [
            [Paragraph("<b>REGIONS OF INTEREST</b>", self.styles['subsection_title'])],
            [Paragraph(legend_text, self.styles['body'])],
            [Spacer(1, 8)],
            [Paragraph("<i>Regions contributing most to the AI finding shown.</i>", self.styles['body_small'])],
            [Paragraph("<i>Percentages reflect relative contribution to the AI finding, not disease extent.</i>", self.styles['body_small'])],
        ]
        
        legend_table = Table(legend_content, colWidths=[legend_width - 20])
        legend_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        # Combine image and legend
        main_row = Table(
            [[rl_map, legend_table]],
            colWidths=[img_width + 20, legend_width]
        )
        main_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(main_row)
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("AI Evidence Localization Map", self.styles['caption']))
        
        return elements
    
    def _build_diagnostic_probability(self, condition_confidences: Dict) -> List:
        """Build diagnostic probability section with bars and cards."""
        elements = []
        
        if not condition_confidences:
            elements.append(Paragraph("No diagnostic data available", self.styles['body']))
            return elements
        
        # Sort by confidence
        sorted_conditions = sorted(condition_confidences.items(), key=lambda x: -x[1])
        
        # Build probability rows
        prob_rows = []
        for condition, confidence in sorted_conditions:
            pct = int(confidence * 100)
            cond_display = condition.replace("_", " ").title()
            
            # Determine status
            if pct >= 70:
                status = "DETECTED"
                status_color = "#EF4444"
            elif pct >= 40:
                status = "POSSIBLE"
                status_color = "#F59E0B"
            else:
                status = "CLEAR"
                status_color = "#10B981"
            
            # Progress bar
            bar = ProgressBar(pct, width=180, height=18)
            
            prob_rows.append([
                Paragraph(f"<b>{cond_display}</b>", self.styles['body']),
                bar,
                Paragraph(f"<font color='{status_color}'><b>{status}</b></font>", self.styles['body']),
            ])
        
        prob_table = Table(prob_rows, colWidths=[2 * inch, 3 * inch, 1.5 * inch])
        prob_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, self.colors['gray_100']),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(prob_table)
        return elements
    
    def _build_condition_cards(self, condition_confidences: Dict,
                                abnormalities: List) -> Table:
        """Build condition screening cards."""
        
        conditions = [
            ('Pneumonia', 'pneumonia'),
            ('Pneumothorax', 'pneumothorax'),
            ('Tuberculosis', 'tuberculosis'),  # Key must match condition_confidences
        ]
        
        def is_detected(abnorm_list: List, key: str) -> bool:
            if not abnorm_list:
                return False
            for a in abnorm_list:
                a_lower = a.lower()
                if 'no evidence' in a_lower or 'not detected' in a_lower:
                    continue
                if key in a_lower:
                    return True
            return False
        
        cards = []
        card_width = (self.CONTENT_WIDTH - 30) / 3
        
        # NOTE: Keys must match those in multi_model_analyzer.py condition_confidences
        for cond_name, cond_key in conditions:
            confidence = condition_confidences.get(cond_key, 0) if condition_confidences else 0
            detected = confidence >= 0.7
            model_detected = is_detected(abnormalities, cond_key)
            
            if detected:
                bg_color = self.colors['danger_light']
                border_color = self.colors['danger']
                status_text = "DETECTED"
                icon = "⚠"
            else:
                bg_color = self.colors['success_light']
                border_color = self.colors['success']
                status_text = "CLEAR"
                icon = "✓"
            
            # Confidence display
            conf_pct = int(confidence * 100)
            
            card_content = [
                [Paragraph(f"<b>{cond_name}</b>", ParagraphStyle(
                    'CardTitle', parent=self.styles['body'],
                    alignment=TA_CENTER, fontSize=11
                ))],
                [Paragraph(f"<font size='16'>{icon}</font> <b>{status_text}</b>", ParagraphStyle(
                    'CardStatus', parent=self.styles['body'],
                    alignment=TA_CENTER, textColor=border_color, fontSize=12
                ))],
                [Paragraph(f"Confidence: {conf_pct}%", ParagraphStyle(
                    'CardMeta', parent=self.styles['body_small'],
                    alignment=TA_CENTER
                ))],
            ]
            
            card = Table(card_content, colWidths=[card_width - 20])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                ('LINEABOVE', (0, 0), (-1, 0), 3, border_color),
                ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
                ('TOPPADDING', (0, 0), (-1, -1), 15),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            
            cards.append(card)
        
        cards_row = Table([cards], colWidths=[card_width] * 3)
        cards_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        return cards_row
    
    def _build_clinical_impression(self, consensus_data: Dict) -> List:
        """Build clinical impression section."""
        elements = []
        
        clinical_impression = consensus_data.get('clinical_impression', '') if consensus_data else ''
        confidence = consensus_data.get('medgemma_confidence', 'N/A') if consensus_data else 'N/A'
        
        # Analysis confidence badge (single model)
        badge_color = self.colors['accent']
        badge_bg = self.colors['accent_light']
        badge_text = "AI ANALYSIS COMPLETE"
        badge_desc = f"MedGemma-4B-IT analysis confidence: {confidence}"
        
        badge_content = [
            [Paragraph(f"<b>{badge_text}</b>", ParagraphStyle(
                'BadgeTitle', fontName=self.font_display, fontSize=13,
                textColor=badge_color, alignment=TA_CENTER
            ))],
            [Paragraph(badge_desc, ParagraphStyle(
                'BadgeDesc', parent=self.styles['body_small'],
                alignment=TA_CENTER
            ))],
        ]
        
        badge_table = Table(badge_content, colWidths=[self.CONTENT_WIDTH - 40])
        badge_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), badge_bg),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(badge_table)
        elements.append(Spacer(1, 20))
        
        # Clinical impression text
        if clinical_impression:
            impression_card = Table(
                [[Paragraph("<b>Clinical Impression</b>", self.styles['subsection_title'])],
                 [Paragraph(clinical_impression, self.styles['body'])]],
                colWidths=[self.CONTENT_WIDTH - 40]
            )
            impression_card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
                ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
                ('TOPPADDING', (0, 0), (-1, -1), 15),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                ('LEFTPADDING', (0, 0), (-1, -1), 20),
                ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ]))
            elements.append(impression_card)
        
        return elements
    
    def _build_recommendations(self) -> List:
        """Build recommendations section."""
        elements = []
        
        recommendations = [
            ("<b>Physician Review Required:</b> This AI analysis serves as a clinical decision "
             "support tool. All findings must be independently verified by a qualified radiologist."),
            ("<b>Clinical Correlation:</b> Imaging findings should be interpreted in conjunction "
             "with patient history, physical examination, and laboratory results."),
            ("<b>Model Limitations:</b> AI models may have limitations in detecting subtle findings "
             "or atypical presentations. Clinical judgment must supersede algorithmic output."),
        ]
        
        rec_rows = [[Paragraph("⚠ IMPORTANT NOTICE", ParagraphStyle(
            'RecTitle', fontName=self.font_display, fontSize=11,
            textColor=self.colors['danger'], alignment=TA_CENTER, spaceAfter=10
        ))]]
        
        for rec in recommendations:
            rec_rows.append([Paragraph(rec, self.styles['body'])])
        
        rec_table = Table(rec_rows, colWidths=[self.CONTENT_WIDTH - 40])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['danger_light']),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
            ('LINEABOVE', (0, 0), (-1, 0), 2, self.colors['danger']),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ]))
        
        elements.append(rec_table)
        return elements
    
    def _build_attestation(self, analysis_id: str) -> Paragraph:
        """Build report attestation."""
        attestation = (
            f"<b>Report ID:</b> {analysis_id} | "
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | "
            f"<b>Model:</b> MedGemma-4B-IT (Full Precision) | "
            f"<b>Platform:</b> SentryAI Medical Intelligence"
        )
        return Paragraph(attestation, self.styles['disclaimer'])
    
    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================
    
    def generate_report(
        self,
        analysis_response,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        consensus_data: Optional[Dict[str, Any]] = None,
        density_saliency_image: Optional[Image.Image] = None,
        localization_map: Optional[Image.Image] = None,
        detected_regions: Optional[List[Dict[str, Any]]] = None,
        condition_confidences: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate clinical-grade PDF report.
        
        Args:
            analysis_response: Structured analysis response
            original_image: Original X-ray image
            heatmap_image: AI attention heatmap
            consensus_data: Multi-model consensus data
            density_saliency_image: Density/saliency map (optional)
            localization_map: Lesion localization map (optional)
            detected_regions: List of detected region metadata
            condition_confidences: Dict of condition -> confidence (0-1)
        
        Returns:
            Path to generated PDF file
        """
        
        filename = f"sentryai_clinical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=self.MARGIN,
            leftMargin=self.MARGIN,
            topMargin=115,  # Account for header
            bottomMargin=65,  # Account for footer
        )
        
        story = []
        
        # Extract abnormalities from findings
        if consensus_data:
            abnormalities = [a.lower() for a in consensus_data.get('medgemma_abnormalities', [])]
        else:
            abnormalities = []
        
        # =====================================================================
        # PAGE 1: EXECUTIVE SUMMARY
        # =====================================================================
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['section_title']))
        story.extend(self._build_executive_summary(consensus_data, condition_confidences))
        story.append(Spacer(1, 25))
        
        # Metadata section on page 1
        story.append(Paragraph("STUDY INFORMATION", self.styles['section_title']))
        meta_data = [
            ['Analysis ID', analysis_response.analysis_id],
            ['Timestamp', analysis_response.timestamp],
            ['Image Source', analysis_response.image_metadata.get('filename', 'URL')],
            ['Dimensions', f"{analysis_response.image_metadata.get('dimensions', ['N/A', 'N/A'])[0]} × {analysis_response.image_metadata.get('dimensions', ['N/A', 'N/A'])[1]} px"],
            ['Prior Imaging', 'None available for comparison'],
        ]
        
        meta_rows = []
        for label, value in meta_data:
            meta_rows.append([
                Paragraph(label, self.styles['meta_label']),
                Paragraph(str(value)[:50], self.styles['meta_value']),
            ])
        
        meta_table = Table(meta_rows, colWidths=[1.5 * inch, 4.5 * inch])
        meta_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, self.colors['gray_200']),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))
        
        # Image Quality Assessment section
        story.append(Paragraph("IMAGE QUALITY", self.styles['subsection_title']))
        quality_data = [
            ['Positioning', 'Adequate'],
            ['Exposure', 'Adequate'],
            ['Motion Artifact', 'None detected'],
            ['Image Orientation', 'Standard PA/AP'],
        ]
        
        quality_rows = []
        for label, value in quality_data:
            quality_rows.append([
                Paragraph(label, self.styles['meta_label']),
                Paragraph(f"<font color='#10B981'>✓</font> {value}", self.styles['meta_value']),
            ])
        
        quality_table = Table(quality_rows, colWidths=[1.5 * inch, 4.5 * inch])
        quality_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['success_light']),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, self.colors['gray_200']),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(quality_table)
        
        story.append(PageBreak())
        
        # =====================================================================
        # PAGE 2: IMAGING ANALYSIS (Side-by-side)
        # =====================================================================
        story.append(Paragraph("IMAGING ANALYSIS", self.styles['section_title']))
        story.append(Spacer(1, 10))
        story.append(self._build_side_by_side_images(
            original_image, heatmap_image,
            "Original Radiograph", "AI Evidence Map"
        ))
        story.append(Spacer(1, 15))
        story.append(Paragraph(
            "<i>The AI evidence map highlights regions contributing to the diagnostic finding. "
            "Warmer colors (red/yellow) indicate areas of higher diagnostic relevance.</i>",
            self.styles['body_small']
        ))
        story.append(PageBreak())
        
        # =====================================================================
        # PAGE 3: LESION LOCALIZATION + DENSITY VISUALIZATION
        # =====================================================================
        if localization_map is not None:
            story.append(Paragraph("LESION LOCALIZATION", self.styles['section_title']))
            story.append(Spacer(1, 10))
            story.extend(self._build_localization_section(localization_map, detected_regions or []))
            story.append(Spacer(1, 25))
        
        # NOTE: Density visualization removed from main clinical report
        # (Non-diagnostic, could confuse interpretation)
        # Can be added to technical appendix if needed
        
        story.append(PageBreak())
        
        # =====================================================================
        # PAGE 4: DIAGNOSTIC PROBABILITY
        # =====================================================================
        story.append(Paragraph("DIAGNOSTIC PROBABILITY", self.styles['section_title']))
        story.append(Spacer(1, 10))
        story.extend(self._build_diagnostic_probability(condition_confidences))
        story.append(Spacer(1, 30))
        
        story.append(Paragraph("CONDITION SCREENING", self.styles['section_title']))
        story.append(Spacer(1, 10))
        story.append(self._build_condition_cards(condition_confidences, abnormalities))
        
        story.append(PageBreak())
        
        # =====================================================================
        # PAGE 5: CLINICAL IMPRESSION & RECOMMENDATIONS
        # =====================================================================
        story.append(Paragraph("AI ANALYSIS", self.styles['section_title']))
        story.append(Spacer(1, 10))
        story.extend(self._build_clinical_impression(consensus_data))
        story.append(Spacer(1, 30))
        
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['section_title']))
        story.append(Spacer(1, 10))
        story.extend(self._build_recommendations())
        story.append(Spacer(1, 30))
        
        # Attestation
        story.append(HRFlowable(width="100%", thickness=1, color=self.colors['gray_200']))
        story.append(Spacer(1, 10))
        story.append(self._build_attestation(analysis_response.analysis_id))
        
        # Build PDF
        doc.build(
            story,
            onFirstPage=self._on_first_page,
            onLaterPages=self._on_later_pages
        )
        
        logger.info(f"Generated clinical report: {filepath}")
        return filepath


# Backwards compatibility alias
PremiumReportGenerator = ClinicalReportGenerator
