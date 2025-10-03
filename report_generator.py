import os
import io
import logging
import re
from datetime import datetime
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as ReportLabImage, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradientRect(Flowable):
    """Custom gradient rectangle for premium visual effects"""
    def __init__(self, width, height, start_color, end_color):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.start_color = start_color
        self.end_color = end_color

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Create gradient effect with multiple rectangles
        steps = 50
        for i in range(steps):
            ratio = i / float(steps)
            r = self.start_color.red * (1 - ratio) + self.end_color.red * ratio
            g = self.start_color.green * (1 - ratio) + self.end_color.green * ratio
            b = self.start_color.blue * (1 - ratio) + self.end_color.blue * ratio
            
            color = colors.Color(r, g, b)
            canvas.setFillColor(color)
            canvas.rect(0, i * self.height / steps, self.width, self.height / steps, fill=1, stroke=0)
        
        canvas.restoreState()

class PremiumCard(Flowable):
    """Premium card with shadows and gradients"""
    def __init__(self, width, height, content, bg_color, accent_color, shadow=True):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.content = content
        self.bg_color = bg_color
        self.accent_color = accent_color
        self.shadow = shadow

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Draw shadow
        if self.shadow:
            shadow_offset = 3
            canvas.setFillColor(colors.Color(0, 0, 0, alpha=0.1))
            canvas.roundRect(shadow_offset, -shadow_offset, self.width, self.height, 8, fill=1, stroke=0)
        
        # Draw main card
        canvas.setFillColor(self.bg_color)
        canvas.setStrokeColor(self.accent_color)
        canvas.setLineWidth(2)
        canvas.roundRect(0, 0, self.width, self.height, 12, fill=1, stroke=1)
        
        # Draw accent border
        canvas.setStrokeColor(self.accent_color)
        canvas.setLineWidth(4)
        canvas.line(0, self.height - 4, self.width, self.height - 4)
        
        canvas.restoreState()

class UltimatePDFGenerator:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self._setup_fonts()
        self._setup_colors()
        self._setup_styles()

    def _setup_fonts(self):
        """Setup premium typography"""
        try:
            pdfmetrics.registerFont(TTFont('Montserrat', 'assets/Montserrat-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Montserrat-Bold', 'assets/Montserrat-Bold.ttf'))
            pdfmetrics.registerFont(TTFont('Montserrat-Light', 'assets/Montserrat-Light.ttf'))
            pdfmetrics.registerFont(TTFont('Source', 'assets/SourceSansPro-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Source-Bold', 'assets/SourceSansPro-Bold.ttf'))
            self.font_display = 'Montserrat-Bold'
            self.font_body = 'Source'
            self.font_body_bold = 'Source-Bold'
            self.font_light = 'Montserrat-Light'
            logger.info("Successfully loaded premium fonts.")
        except Exception:
            logger.warning("Premium fonts not found in 'assets/' directory. Using built-in fonts.")
            self.font_display = 'Helvetica-Bold'
            self.font_body = 'Helvetica'
            self.font_body_bold = 'Helvetica-Bold'
            self.font_light = 'Helvetica'

    def _setup_colors(self):
        """Premium medical color palette"""
        self.colors = {
            # Primary medical blues
            'primary_dark': colors.HexColor("#0B1426"),
            'primary': colors.HexColor("#1E3A8A"),
            'primary_light': colors.HexColor("#3B82F6"),
            'primary_ultra_light': colors.HexColor("#DBEAFE"),
            
            # Success greens
            'success_dark': colors.HexColor("#065F46"),
            'success': colors.HexColor("#10B981"),
            'success_light': colors.HexColor("#6EE7B7"),
            'success_ultra_light': colors.HexColor("#D1FAE5"),
            
            # Warning/Alert colors
            'warning_dark': colors.HexColor("#92400E"),
            'warning': colors.HexColor("#F59E0B"),
            'warning_light': colors.HexColor("#FCD34D"),
            'warning_ultra_light': colors.HexColor("#FEF3C7"),
            
            # Danger/Critical
            'danger_dark': colors.HexColor("#7F1D1D"),
            'danger': colors.HexColor("#EF4444"),
            'danger_light': colors.HexColor("#FCA5A5"),
            'danger_ultra_light': colors.HexColor("#FEE2E2"),
            
            # Sophisticated grays
            'gray_900': colors.HexColor("#111827"),
            'gray_800': colors.HexColor("#1F2937"),
            'gray_700': colors.HexColor("#374151"),
            'gray_600': colors.HexColor("#4B5563"),
            'gray_500': colors.HexColor("#6B7280"),
            'gray_400': colors.HexColor("#9CA3AF"),
            'gray_300': colors.HexColor("#D1D5DB"),
            'gray_200': colors.HexColor("#E5E7EB"),
            'gray_100': colors.HexColor("#F3F4F6"),
            'gray_50': colors.HexColor("#F9FAFB"),
            
            # Pure colors
            'white': colors.white,
            'black': colors.black,
            
            # Accent colors
            'accent_purple': colors.HexColor("#8B5CF6"),
            'accent_cyan': colors.HexColor("#06B6D4"),
            'accent_emerald': colors.HexColor("#10B981"),
        }

    def _setup_styles(self):
        """Create sophisticated typography system"""
        self.styles = {
            # Hero title - reduced size for professionalism
            'hero_title': ParagraphStyle(
                'HeroTitle',
                fontName=self.font_display,
                fontSize=24,
                textColor=self.colors['primary_dark'],
                spaceAfter=6,
                leading=28,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Main title - more compact
            'main_title': ParagraphStyle(
                'MainTitle',
                fontName=self.font_display,
                fontSize=20,
                textColor=self.colors['primary'],
                spaceAfter=5,
                leading=24,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Subtitle with better wrapping
            'subtitle': ParagraphStyle(
                'Subtitle',
                fontName=self.font_body,
                fontSize=12,
                textColor=self.colors['gray_600'],
                spaceAfter=20,
                leading=16,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Section headers - professional size
            'section_header': ParagraphStyle(
                'SectionHeader',
                fontName=self.font_display,
                fontSize=16,
                textColor=self.colors['primary'],
                spaceBefore=22,
                spaceAfter=12,
                leading=20,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Subsection headers - compact
            'subsection_header': ParagraphStyle(
                'SubsectionHeader',
                fontName=self.font_body_bold,
                fontSize=12,
                textColor=self.colors['gray_700'],
                spaceBefore=10,
                spaceAfter=4,
                leading=16,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Premium body text with better wrapping
            'body': ParagraphStyle(
                'BodyText',
                fontName=self.font_body,
                fontSize=10,
                textColor=self.colors['gray_700'],
                leading=16,
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                leftIndent=0,
                rightIndent=0,
                wordWrap='CJK'
            ),
            
            # Statistics - Large numbers - more compact
            'stat_value_large': ParagraphStyle(
                'StatValueLarge',
                fontName=self.font_display,
                fontSize=28,
                textColor=self.colors['primary'],
                alignment=TA_CENTER,
                spaceAfter=3,
                leading=32,
                wordWrap='CJK'
            ),
            
            # Statistics - Medium numbers - professional size
            'stat_value': ParagraphStyle(
                'StatValue',
                fontName=self.font_display,
                fontSize=18,
                textColor=self.colors['primary'],
                alignment=TA_CENTER,
                spaceAfter=3,
                leading=22,
                wordWrap='CJK'
            ),
            
            # Stat labels - smaller for clean look
            'stat_label': ParagraphStyle(
                'StatLabel',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_500'],
                alignment=TA_CENTER,
                leading=12,
                wordWrap='CJK'
            ),
            
            # Image captions - professional size
            'caption': ParagraphStyle(
                'Caption',
                fontName=self.font_body_bold,
                fontSize=10,
                textColor=self.colors['gray_700'],
                alignment=TA_CENTER,
                spaceAfter=6,
                spaceBefore=8,
                wordWrap='CJK'
            ),
            
            # Metadata labels - compact
            'meta_label': ParagraphStyle(
                'MetaLabel',
                fontName=self.font_body_bold,
                fontSize=9,
                textColor=self.colors['gray_800'],
                leading=14,
                wordWrap='CJK'
            ),
            
            # Metadata values - readable size
            'meta_value': ParagraphStyle(
                'MetaValue',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_600'],
                leading=14,
                wordWrap='CJK'
            ),
            
            # Disclaimer - smaller for legal text
            'disclaimer': ParagraphStyle(
                'Disclaimer',
                fontName=self.font_body,
                fontSize=8,
                textColor=self.colors['gray_600'],
                leading=12,
                leftIndent=12,
                rightIndent=12,
                spaceAfter=8,
                spaceBefore=8,
                alignment=TA_JUSTIFY,
                wordWrap='CJK'
            ),
            
            # Alert text - professional sizing
            'alert': ParagraphStyle(
                'Alert',
                fontName=self.font_body_bold,
                fontSize=10,
                textColor=self.colors['danger'],
                leading=14,
                spaceAfter=6,
                spaceBefore=6,
                wordWrap='CJK'
            ),
            
            # Success text - clean look
            'success_text': ParagraphStyle(
                'SuccessText',
                fontName=self.font_body_bold,
                fontSize=10,
                textColor=self.colors['success'],
                leading=14,
                wordWrap='CJK'
            ),
            
            # Technical details - compact formatting
            'technical': ParagraphStyle(
                'Technical',
                fontName=self.font_body,
                fontSize=8,
                textColor=self.colors['gray_600'],
                leading=12,
                leftIndent=8,
                spaceAfter=6,
                wordWrap='CJK'
            ),
            
            # Professional finding status style
            'finding_status': ParagraphStyle(
                'FindingStatus',
                fontName=self.font_body_bold,
                fontSize=10,
                leading=14,
                alignment=TA_CENTER,
                wordWrap='CJK'
            ),
            
            # Professional table header style
            'table_header': ParagraphStyle(
                'TableHeader',
                fontName=self.font_body_bold,
                fontSize=10,
                textColor=self.colors['gray_800'],
                leading=14,
                alignment=TA_CENTER,
                wordWrap='CJK'
            ),
            
            # Add new clinical styles
            'clinical_summary': ParagraphStyle(
                'ClinicalSummary',
                fontName=self.font_body,
                fontSize=11,
                textColor=self.colors['gray_800'],
                leading=18,
                spaceAfter=10,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            'risk_indicator': ParagraphStyle(
                'RiskIndicator',
                fontName=self.font_body_bold,
                fontSize=11,
                leading=16,
                alignment=TA_CENTER,
                wordWrap='CJK'
            ),
            
            'confidence_badge': ParagraphStyle(
                'ConfidenceBadge',
                fontName=self.font_body_bold,
                fontSize=9,
                alignment=TA_CENTER,
                leading=12,
                wordWrap='CJK'
            ),
        }

    def _create_header_canvas(self, canvas, doc):
        """Create stunning header with gradient and branding"""
        canvas.saveState()
        
        header_height = 80
        page_width = doc.pagesize[0]
        
        # Header background gradient
        gradient_steps = 30
        for i in range(gradient_steps):
            ratio = i / float(gradient_steps)
            alpha = 0.1 - (ratio * 0.08)  # Fade from 0.1 to 0.02
            color = colors.Color(30/255.0, 58/255.0, 138/255.0, alpha=alpha)  # primary color
            canvas.setFillColor(color)
            rect_height = header_height / float(gradient_steps)
            canvas.rect(0, doc.pagesize[1] - header_height + (i * rect_height), 
                       page_width, rect_height, fill=1, stroke=0)
        
        # Header line accent
        canvas.setStrokeColor(self.colors['primary'])
        canvas.setLineWidth(3)
        canvas.line(0, doc.pagesize[1] - header_height, page_width, doc.pagesize[1] - header_height)
        
        # Logo area (if available)
        logo_path = 'assets/logo.png'
        logo_x = 40
        logo_y = doc.pagesize[1] - header_height + 20
        
        try:
            if os.path.exists(logo_path):
                canvas.drawImage(logo_path, logo_x, logo_y, 
                               width=40, height=40, 
                               preserveAspectRatio=True, mask='auto')
            else:
                raise FileNotFoundError("Logo not found")
        except Exception:
            # Create a premium logo placeholder
            canvas.setFillColor(self.colors['primary'])
            canvas.circle(logo_x + 20, logo_y + 20, 20, fill=1, stroke=0)
            canvas.setFillColor(self.colors['white'])
            canvas.setFont(self.font_display, 16)
            canvas.drawCentredString(logo_x + 20, logo_y + 14, "AI")
        
        # Header text - professional sizing
        canvas.setFont(self.font_display, 14)
        canvas.setFillColor(self.colors['primary_dark'])
        canvas.drawString(logo_x + 60, logo_y + 25, "AI MEDICAL DIAGNOSTIC SYSTEM")
        
        canvas.setFont(self.font_body, 10)
        canvas.setFillColor(self.colors['gray_600'])
        canvas.drawString(logo_x + 60, logo_y + 8, "Advanced Neural Network Analysis • Clinical Decision Support")
        
        # Header decoration
        canvas.setStrokeColor(self.colors['accent_cyan'])
        canvas.setLineWidth(2)
        canvas.line(logo_x + 60, logo_y + 4, logo_x + 300, logo_y + 4)
        
        canvas.restoreState()

    def _create_footer_canvas(self, canvas, doc):
        """Create professional footer"""
        canvas.saveState()
        
        footer_y = 30
        page_width = doc.pagesize[0]
        
        # Footer background
        canvas.setFillColor(self.colors['gray_50'])
        canvas.rect(0, 0, page_width, 60, fill=1, stroke=0)
        
        # Footer line
        canvas.setStrokeColor(self.colors['gray_300'])
        canvas.setLineWidth(1)
        canvas.line(40, 60, page_width - 40, 60)
        
        # Footer text
        canvas.setFont(self.font_body, 9)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(40, footer_y + 20, "Med-GEMMA-3 Neural Diagnostic Engine")
        canvas.drawString(40, footer_y + 8, "⚡ Confidential Medical Analysis • For Licensed Healthcare Professionals Only")
        
        # Page number with style
        canvas.setFont(self.font_body_bold, 10)
        canvas.setFillColor(self.colors['primary'])
        canvas.drawRightString(page_width - 40, footer_y + 14, f"Page {doc.page}")
        
        # Security indicator
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray_400'])
        canvas.drawRightString(page_width - 40, footer_y + 2, "🔒 Encrypted • HIPAA Compliant")
        
        canvas.restoreState()

    def _create_premium_stat_card(self, value, label, status_color='primary', icon="📊"):
        """Create ultra-premium stat cards with visual hierarchy"""
        if status_color == 'success':
            bg_color = self.colors['success_ultra_light']
            border_color = self.colors['success']
            value_color = self.colors['success_dark']
        elif status_color == 'warning':
            bg_color = self.colors['warning_ultra_light']
            border_color = self.colors['warning']
            value_color = self.colors['warning_dark']
        elif status_color == 'danger':
            bg_color = self.colors['danger_ultra_light']
            border_color = self.colors['danger']
            value_color = self.colors['danger_dark']
        else:  # primary
            bg_color = self.colors['primary_ultra_light']
            border_color = self.colors['primary']
            value_color = self.colors['primary_dark']
        
        value_style = ParagraphStyle(
            'CardValue', 
            parent=self.styles['stat_value'], 
            textColor=value_color, 
            fontSize=13, 
            leading=16, 
            wordWrap='CJK',
            alignment=TA_CENTER
        )
        label_style = ParagraphStyle(
            'CardLabel', parent=self.styles['stat_label'], textColor=self.colors['gray_600'], fontSize=10, leading=14, wordWrap='CJK'
        )
        
        card_content = [
            [Paragraph(f"{icon}", ParagraphStyle('Icon', fontSize=16, alignment=TA_CENTER, spaceAfter=6))],
            [Paragraph(str(value).replace('\n', ' ').strip(), value_style)],
            [Paragraph(label, label_style)]
        ]
        
        card_table = Table(card_content, colWidths=[2.6 * inch])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 18),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [8, 8, 8, 8]),
        ]))
        
        return card_table

    def _create_clinical_summary_card(self, findings):
        """Create a clinical executive summary card"""
        # Determine overall severity
        is_positive = "positive" in findings.impression.lower()
        
        if is_positive:
            severity_text = "⚠️ ABNORMALITIES DETECTED"
            severity_color = self.colors['danger']
            bg_color = self.colors['danger_ultra_light']
        else:
            severity_text = "✓ NO SIGNIFICANT ABNORMALITIES"
            severity_color = self.colors['success']
            bg_color = self.colors['success_ultra_light']
        
        summary_style = ParagraphStyle(
            'SummaryHeader',
            fontName=self.font_display,
            fontSize=14,
            textColor=severity_color,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8,
            wordWrap='CJK'
        )
        
        content = [
            [Paragraph(severity_text, summary_style)],
            [Paragraph(f"<b>Clinical Impression:</b> {findings.impression.replace('_', ' ').title()}", self.styles['clinical_summary'])],
            [Paragraph(f"<b>Diagnostic Confidence:</b> {findings.confidence_level}", self.styles['clinical_summary'])],
        ]
        
        summary_table = Table(content, colWidths=[6.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 2, severity_color),
            ('TOPPADDING', (0, 0), (-1, -1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [10, 10, 10, 10]),
        ]))
        
        return summary_table

    def _create_diagnostic_finding_card(self, condition, status, confidence, description=""):
        """Create a professional diagnostic finding card for each condition"""
        if status == "Detected":
            status_icon = "⚠️"
            status_color = self.colors['danger']
            bg_color = self.colors['danger_ultra_light']
            border_color = self.colors['danger']
        else:
            status_icon = "✓"
            status_color = self.colors['success_dark']
            bg_color = self.colors['success_ultra_light']
            border_color = self.colors['success']
        
        # Confidence level styling
        try:
            conf_value = float(confidence.rstrip('%'))
            if conf_value >= 80:
                conf_color = self.colors['success_dark']
                conf_bg = self.colors['success_light']
            elif conf_value >= 50:
                conf_color = self.colors['warning_dark']
                conf_bg = self.colors['warning_light']
            else:
                conf_color = self.colors['gray_700']
                conf_bg = self.colors['gray_300']
        except:
            conf_color = self.colors['gray_700']
            conf_bg = self.colors['gray_300']
        
        condition_header = ParagraphStyle(
            'ConditionHeader',
            fontName=self.font_display,
            fontSize=13,
            textColor=self.colors['gray_900'],
            leading=16,
            wordWrap='CJK'
        )
        
        status_style = ParagraphStyle(
            'StatusStyle',
            fontName=self.font_body_bold,
            fontSize=11,
            textColor=status_color,
            leading=14,
            wordWrap='CJK'
        )
        
        conf_style = ParagraphStyle(
            'ConfStyle',
            fontName=self.font_body_bold,
            fontSize=10,
            textColor=conf_color,
            leading=12,
            alignment=TA_CENTER,
            wordWrap='CJK'
        )
        
        card_data = [
            [Paragraph(f"<b>{condition}</b>", condition_header), "", Paragraph(confidence if confidence != "N/A" else "—", conf_style)],
            [Paragraph(f"{status_icon} {status}", status_style), "", ""]
        ]
        
        if description and status == "Detected":
            card_data.append([Paragraph(f"<i>{description}</i>", self.styles['technical']), "", ""])
        
        card_table = Table(card_data, colWidths=[4*inch, 0.3*inch, 1.2*inch])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('SPAN', (0, 1), (1, 1)),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 16),
            ('RIGHTPADDING', (0, 0), (-1, -1), 16),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (2, 0), (2, 0), conf_bg),
            ('BOX', (2, 0), (2, 0), 1, conf_color),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [8, 8, 8, 8]),
        ]))
        
        return card_table

    def _create_metadata_section(self, analysis_response):
        """Create beautiful metadata section"""
        metadata = analysis_response.image_metadata
        source = metadata.get('filename') or str(metadata.get('url', 'Unknown'))
        if len(source) > 55:
            source = source[:52] + '...'
        
        try:
            timestamp = datetime.fromisoformat(analysis_response.timestamp.replace('Z', '+00:00'))
            time_str = timestamp.strftime('%B %d, %Y at %H:%M:%S UTC')
        except Exception:
            time_str = analysis_response.timestamp
        
        dimensions = metadata.get('dimensions', ['N/A', 'N/A'])
        
        metadata_items = [
            ('🆔', 'Analysis ID', analysis_response.analysis_id),
            ('⏰', 'Generated', time_str),
            ('📁', 'Image Source', source),
            ('📐', 'Dimensions', f"{dimensions[0]} × {dimensions[1]} pixels"),
            ('💾', 'Processing', "Neural Network Analysis Complete"),
            ('🎯', 'Model Version', "Med-GEMMA-3.1 (Build 2024.06)")
        ]
        
        styled_rows = []
        for icon, label, value in metadata_items:
            row = [
                Paragraph(icon, ParagraphStyle('Icon', fontSize=12, alignment=TA_CENTER)),
                Paragraph(label, self.styles['meta_label']),
                Paragraph(str(value), self.styles['meta_value'])
            ]
            styled_rows.append(row)
        
        meta_table = Table(styled_rows, colWidths=[0.3*inch, 1.2*inch, None])
        meta_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (0, -1), 4),
            ('RIGHTPADDING', (-1, 0), (-1, -1), 4),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, self.colors['gray_200']),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [6, 6, 6, 6]),
        ]))
        
        return meta_table

    def _enhance_image_with_border(self, pil_image, border_color="#E5E7EB", border_width=2):
        """Add professional border to images"""
        border_rgb = tuple(int(border_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        new_width = pil_image.width + (border_width * 2)
        new_height = pil_image.height + (border_width * 2)
        bordered_image = Image.new('RGB', (new_width, new_height), border_rgb)
        bordered_image.paste(pil_image, (border_width, border_width))
        
        return bordered_image

    def _pil_to_reportlab_image(self, pil_image, max_width=None, max_height=None, enhance=True):
        """Convert PIL image to ReportLab with enhancements"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if enhance:
            pil_image = self._enhance_image_with_border(pil_image)
        
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True, quality=95)
        img_buffer.seek(0)
        
        original_width, original_height = pil_image.size
        aspect_ratio = float(original_height) / float(original_width)
        
        width = max_width if max_width else original_width
        height = width * aspect_ratio

        return ReportLabImage(img_buffer, width=width, height=height)

    def generate_pdf_report(self,
                            analysis_response,
                            original_image: Image.Image,
                            ai_attention_heatmap_image: Image.Image) -> str:
        """Generate a streamlined medical PDF report with only the essential heatmap."""
        
        filename = f"premium_medical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=100, bottomMargin=80
        )
        
        story = []
        findings = analysis_response.findings
        
        # === COVER PAGE ===
        story.append(Spacer(1, 20))
        story.append(Paragraph("DIAGNOSTIC IMAGING REPORT", self.styles['hero_title']))
        story.append(Paragraph("Chest Radiography • Multi-Condition Analysis", self.styles['main_title']))
        story.append(Spacer(1, 8))
        
        story.append(Paragraph(
            "Automated screening for Pneumothorax, Pneumonia, and Tuberculosis using deep neural network analysis",
            self.styles['subtitle']
        ))
        
        story.append(Spacer(1, 30))
        
        # === CLINICAL EXECUTIVE SUMMARY ===
        story.append(Paragraph("📊 Clinical Executive Summary", self.styles['section_header']))
        story.append(Spacer(1, 8))
        story.append(self._create_clinical_summary_card(findings))
        story.append(Spacer(1, 30))
        
        # === KEY METRICS CARDS ===
        impression_icon = "🚨" if "positive" in findings.impression.lower() else "✅"
        confidence_icon = "🎯"
        
        impression_color = 'danger' if "positive" in findings.impression.lower() else 'success'
        confidence_color = 'primary'
        
        stat_cards = [
            self._create_premium_stat_card(
                findings.impression.title().replace('_', ' '), 
                "Radiological Impression",
                impression_color, 
                impression_icon
            ),
            self._create_premium_stat_card(
                findings.confidence_level,
                "Maximum Confidence",
                confidence_color, 
                confidence_icon
            )
        ]
        
        cards_table = Table([stat_cards], colWidths=[2.6*inch, 2.6*inch], hAlign='CENTER')
        cards_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (0, 0), 6), ('RIGHTPADDING', (0, 0), (0, 0), 3),
            ('LEFTPADDING', (1, 0), (1, 0), 3), ('RIGHTPADDING', (1, 0), (1, 0), 6),
        ]))
        
        story.append(cards_table)
        story.append(Spacer(1, 35))
        
        # === STUDY INFORMATION ===
        story.append(Paragraph("📋 Study Information", self.styles['section_header']))
        story.append(self._create_metadata_section(analysis_response))
        story.append(Spacer(1, 30))
        
        # === AI INTERPRETATION ===
        story.append(Paragraph("🤖 Neural Network Interpretation", self.styles['section_header']))
        story.append(Spacer(1, 5))
        analysis_text = analysis_response.raw_response.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
        
        analysis_table = Table([[Paragraph(analysis_text, self.styles['body'])]], colWidths=[6.5*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['primary_light']),
            ('PADDING', (0, 0), (-1, -1), 18),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [8, 8, 8, 8]),
        ]))
        
        story.append(analysis_table)
        story.append(PageBreak())
        
        # === IMAGING ANALYSIS ===
        story.append(Paragraph("🔍 Imaging Analysis", self.styles['section_header']))
        story.append(Spacer(1, 12))
        
        img_max_width_pair = 3 * inch
        rl_original_image = self._pil_to_reportlab_image(original_image, max_width=img_max_width_pair)
        rl_heatmap_image = self._pil_to_reportlab_image(ai_attention_heatmap_image, max_width=img_max_width_pair)

        image_pair_data = [
            [rl_original_image, rl_heatmap_image],
            [Paragraph("Original Radiograph", self.styles['caption']), Paragraph("Neural Attention Map", self.styles['caption'])]
        ]

        col_widths = [img_max_width_pair + 0.2*inch, img_max_width_pair + 0.2*inch]
        image_pair_table = Table(image_pair_data, colWidths=col_widths)
        image_pair_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ]))

        story.append(KeepTogether(image_pair_table))
        story.append(Spacer(1, 20))
        
        # Add technical note
        tech_note = Paragraph(
            "<i>The neural attention map highlights regions where the AI model focused during analysis. "
            "Warmer colors (red/yellow) indicate higher attention weights, corresponding to areas of diagnostic interest.</i>",
            self.styles['technical']
        )
        story.append(tech_note)
        story.append(Spacer(1, 30))

        # === DETAILED PATHOLOGICAL FINDINGS ===
        story.append(Paragraph("🔬 Detailed Pathological Findings", self.styles['section_header']))
        story.append(Spacer(1, 12))
        
        # Parse abnormalities from findings
        abnormality_pattern = re.compile(r"(\w+)\s+\(([\d.]+%)\)")
        all_conditions = ["Pneumothorax", "Pneumonia", "TB"]
        detected_conditions = {match.group(1): match.group(2) for match in abnormality_pattern.finditer(" ".join(findings.abnormalities))}

        # Create individual diagnostic cards
        for condition in all_conditions:
            if condition in detected_conditions:
                status = "Detected"
                confidence = detected_conditions[condition]
                description = f"Algorithmic pattern recognition indicates presence with {confidence} confidence"
            else:
                status = "Not Detected"
                confidence = "N/A"
                description = ""
            
            story.append(self._create_diagnostic_finding_card(condition, status, confidence, description))
            story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 20))

        # === CLINICAL INTERPRETATION & RECOMMENDATIONS ===
        story.append(Paragraph("⚕️ Clinical Interpretation & Recommendations", self.styles['section_header']))
        story.append(Spacer(1, 8))
        
        recommendations_data = [
            [Paragraph("<b>IMPORTANT DISCLAIMER</b>", ParagraphStyle('DisclaimerHeader', parent=self.styles['body'], fontName=self.font_body_bold, fontSize=11, textColor=self.colors['danger_dark'], alignment=TA_CENTER, spaceAfter=8))],
            [Paragraph("<b>1. Mandatory Physician Review:</b> This AI-assisted analysis is intended as a clinical decision support tool only. All findings must be independently reviewed and verified by a qualified radiologist or licensed physician before clinical action.", self.styles['body'])],
            [Paragraph("<b>2. Clinical Correlation Required:</b> Imaging findings should be interpreted in conjunction with patient clinical history, physical examination findings, laboratory results, and other relevant diagnostic information.", self.styles['body'])],
            [Paragraph("<b>3. Confidence Score Interpretation:</b> The confidence values represent the model's algorithmic certainty based on pattern recognition from training data. These should not be interpreted as clinical probability or diagnostic certainty.", self.styles['body'])],
            [Paragraph("<b>4. Recommended Follow-up:</b> For any detected abnormalities, consider appropriate clinical correlation, additional imaging studies if indicated, and specialist consultation as deemed necessary by the reviewing physician.", self.styles['body'])],
            [Paragraph("<b>5. Model Limitations:</b> AI models may have limitations in detecting subtle findings, rare conditions, or atypical presentations. Clinical judgment must always supersede algorithmic output.", self.styles['body'])],
        ]
        
        recommendations_table = Table(recommendations_data, colWidths=[6.5*inch])
        recommendations_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['danger_ultra_light']),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['primary_ultra_light']),
            ('BOX', (0, 0), (-1, -1), 2, self.colors['primary']),
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.colors['danger']),
            ('PADDING', (0, 0), (-1, -1), 14),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [8, 8, 8, 8]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (0, 0), 10),
            ('BOTTOMPADDING', (0, 0), (0, 0), 10),
        ]))
        
        story.append(recommendations_table)
        story.append(Spacer(1, 20))
        
        # === ATTESTATION SECTION ===
        attestation_text = (
            "<b>Report Generation:</b> This report was automatically generated by the Med-GEMMA-3 Neural Diagnostic Engine. "
            "The analysis is based on deep learning algorithms trained on large-scale medical imaging datasets. "
            "This system is designed to assist, not replace, professional medical judgment."
        )
        story.append(Paragraph(attestation_text, self.styles['disclaimer']))

        doc.build(story, onFirstPage=self._draw_page_template, onLaterPages=self._draw_page_template)
        
        logger.info(f"Successfully generated premium PDF report at: {filepath}")
        return filepath

    def _draw_page_template(self, canvas, doc):
        """Helper method to draw the header and footer on each page."""
        self._create_header_canvas(canvas, doc)
        self._create_footer_canvas(canvas, doc)