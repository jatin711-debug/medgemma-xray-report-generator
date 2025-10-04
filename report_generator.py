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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as ReportLabImage, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
import math

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

class ConfidenceGauge(Flowable):
    """DeepMind-style confidence gauge visualization"""
    def __init__(self, confidence_value, width=180, height=120):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        try:
            self.value = float(str(confidence_value).rstrip('%'))
        except:
            self.value = 0
    
    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Center point
        cx = self.width / 2
        cy = 40
        radius = 35
        
        # Background arc
        canvas.setStrokeColor(colors.HexColor("#E8EAF6"))
        canvas.setLineWidth(8)
        canvas.setLineCap(1)
        
        # Draw background semicircle
        path = canvas.beginPath()
        for angle in range(180, 360, 2):
            rad = math.radians(angle)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            if angle == 180:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        canvas.drawPath(path, stroke=1, fill=0)
        
        # Confidence arc
        end_angle = 180 + (self.value / 100.0) * 180
        
        if self.value >= 80:
            arc_color = colors.HexColor("#4CAF50")
        elif self.value >= 60:
            arc_color = colors.HexColor("#2196F3")
        elif self.value >= 40:
            arc_color = colors.HexColor("#FF9800")
        else:
            arc_color = colors.HexColor("#F44336")
        
        canvas.setStrokeColor(arc_color)
        canvas.setLineWidth(8)
        
        path2 = canvas.beginPath()
        for angle_deg in range(180, int(end_angle), 2):
            rad = math.radians(angle_deg)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            if angle_deg == 180:
                path2.moveTo(x, y)
            else:
                path2.lineTo(x, y)
        canvas.drawPath(path2, stroke=1, fill=0)
        
        # Center value
        canvas.setFillColor(colors.HexColor("#1A237E"))
        canvas.setFont("Helvetica-Bold", 22)
        canvas.drawCentredString(cx, cy - 5, f"{int(self.value)}%")
        
        # Label
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#616161"))
        canvas.drawCentredString(cx, cy - 25, "CONFIDENCE")
        
        canvas.restoreState()

class RiskMatrix(Flowable):
    """DeepMind-style risk assessment matrix"""
    def __init__(self, conditions_data, width=280, height=140):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.conditions = conditions_data
    
    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Title
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(colors.HexColor("#37474F"))
        canvas.drawString(0, self.height - 15, "MULTI-CONDITION RISK ASSESSMENT")
        
        bar_height = 20
        bar_spacing = 8
        start_y = self.height - 45
        
        for i, (condition, confidence) in enumerate(self.conditions):
            y_pos = start_y - (i * (bar_height + bar_spacing))
            
            try:
                conf_val = float(str(confidence).rstrip('%'))
            except:
                conf_val = 0
            
            # Background bar
            canvas.setFillColor(colors.HexColor("#ECEFF1"))
            canvas.rect(100, y_pos, 170, bar_height, fill=1, stroke=0)
            
            # Confidence bar
            bar_width = (conf_val / 100.0) * 170
            if conf_val >= 70:
                bar_color = colors.HexColor("#EF5350")
            elif conf_val >= 40:
                bar_color = colors.HexColor("#FFA726")
            else:
                bar_color = colors.HexColor("#66BB6A")
            
            canvas.setFillColor(bar_color)
            canvas.rect(100, y_pos, bar_width, bar_height, fill=1, stroke=0)
            
            # Condition label
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.HexColor("#37474F"))
            canvas.drawString(0, y_pos + 6, condition)
            
            # Percentage
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.HexColor("#FFFFFF") if conf_val > 30 else colors.HexColor("#37474F"))
            canvas.drawString(105, y_pos + 6, f"{int(conf_val)}%")
        
        canvas.restoreState()

class DeepMindMetricCard(Flowable):
    """Clean, modern metric card in DeepMind style"""
    def __init__(self, value, label, sublabel="", width=140, height=90):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.value = value
        self.label = label
        self.sublabel = sublabel
    
    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Subtle background
        canvas.setFillColor(colors.HexColor("#FAFAFA"))
        canvas.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        
        # Left accent line
        canvas.setFillColor(colors.HexColor("#1976D2"))
        canvas.rect(0, 0, 3, self.height, fill=1, stroke=0)
        
        # Value
        canvas.setFont("Helvetica-Bold", 24)
        canvas.setFillColor(colors.HexColor("#1A237E"))
        
        # Handle multi-line values
        value_str = str(self.value).replace('\n', ' ').strip()
        if len(value_str) > 15:
            canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(self.width/2, self.height - 40, value_str)
        
        # Label
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#616161"))
        canvas.drawCentredString(self.width/2, self.height - 60, self.label.upper())
        
        # Sublabel
        if self.sublabel:
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.HexColor("#9E9E9E"))
            canvas.drawCentredString(self.width/2, self.height - 75, self.sublabel)
        
        canvas.restoreState()

# ...existing code...

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
        """DeepMind-inspired color palette"""
        self.colors = {
            # Primary - DeepMind blues
            'deepmind_primary': colors.HexColor("#1A237E"),
            'deepmind_blue': colors.HexColor("#1976D2"),
            'deepmind_light_blue': colors.HexColor("#64B5F6"),
            'deepmind_ultra_light': colors.HexColor("#E3F2FD"),
            
            # Scientific greens
            'science_green': colors.HexColor("#388E3C"),
            'success': colors.HexColor("#66BB6A"),
            'success_light': colors.HexColor("#C8E6C9"),
            'success_ultra_light': colors.HexColor("#F1F8E9"),
            
            # Alert colors - muted
            'alert_red': colors.HexColor("#E53935"),
            'alert_orange': colors.HexColor("#FB8C00"),
            'alert_red_light': colors.HexColor("#FFCDD2"),
            'alert_orange_light': colors.HexColor("#FFE0B2"),
            
            # Modern grays - more contrast
            'gray_900': colors.HexColor("#212121"),
            'gray_800': colors.HexColor("#424242"),
            'gray_700': colors.HexColor("#616161"),
            'gray_600': colors.HexColor("#757575"),
            'gray_500': colors.HexColor("#9E9E9E"),
            'gray_400': colors.HexColor("#BDBDBD"),
            'gray_300': colors.HexColor("#E0E0E0"),
            'gray_200': colors.HexColor("#EEEEEE"),
            'gray_100': colors.HexColor("#F5F5F5"),
            'gray_50': colors.HexColor("#FAFAFA"),
            
            # Accent colors
            'accent_purple': colors.HexColor("#7E57C2"),
            'accent_teal': colors.HexColor("#26A69A"),
            'accent_amber': colors.HexColor("#FFA726"),
            
            # Base colors
            'white': colors.white,
            'black': colors.black,
            
            # Legacy mappings for compatibility
            'primary_dark': colors.HexColor("#1A237E"),
            'primary': colors.HexColor("#1976D2"),
            'primary_light': colors.HexColor("#64B5F6"),
            'primary_ultra_light': colors.HexColor("#E3F2FD"),
            'success_dark': colors.HexColor("#388E3C"),
            'warning': colors.HexColor("#FB8C00"),
            'warning_dark': colors.HexColor("#EF6C00"),
            'warning_light': colors.HexColor("#FFB74D"),
            'warning_ultra_light': colors.HexColor("#FFE0B2"),
            'danger': colors.HexColor("#E53935"),
            'danger_dark': colors.HexColor("#C62828"),
            'danger_light': colors.HexColor("#EF5350"),
            'danger_ultra_light': colors.HexColor("#FFCDD2"),
        }

    def _setup_styles(self):
        """DeepMind-inspired typography system"""
        self.styles = {
            # Display titles - refined sizing
            'hero_title': ParagraphStyle(
                'HeroTitle',
                fontName=self.font_display,
                fontSize=28,
                textColor=self.colors['deepmind_primary'],
                spaceAfter=4,
                leading=34,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            'main_title': ParagraphStyle(
                'MainTitle',
                fontName=self.font_body,
                fontSize=14,
                textColor=self.colors['gray_600'],
                spaceAfter=8,
                leading=18,
                alignment=TA_LEFT,
                wordWrap='CJK',
                letterSpacing=0.5
            ),
            
            'subtitle': ParagraphStyle(
                'Subtitle',
                fontName=self.font_body,
                fontSize=11,
                textColor=self.colors['gray_500'],
                spaceAfter=24,
                leading=16,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Section headers - modern spacing
            'section_header': ParagraphStyle(
                'SectionHeader',
                fontName=self.font_display,
                fontSize=14,
                textColor=self.colors['deepmind_primary'],
                spaceBefore=30,
                spaceAfter=16,
                leading=18,
                alignment=TA_LEFT,
                wordWrap='CJK',
                letterSpacing=0.3
            ),
            
            'subsection_header': ParagraphStyle(
                'SubsectionHeader',
                fontName=self.font_body_bold,
                fontSize=11,
                textColor=self.colors['gray_700'],
                spaceBefore=12,
                spaceAfter=6,
                leading=15,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Body text - optimized readability
            'body': ParagraphStyle(
                'BodyText',
                fontName=self.font_body,
                fontSize=10,
                textColor=self.colors['gray_700'],
                leading=17,
                spaceAfter=10,
                alignment=TA_LEFT,
                leftIndent=0,
                rightIndent=0,
                wordWrap='CJK'
            ),
            
            # Technical text
            'technical': ParagraphStyle(
                'Technical',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_600'],
                leading=14,
                leftIndent=0,
                spaceAfter=8,
                wordWrap='CJK'
            ),
            
            # Modern caption style
            'caption': ParagraphStyle(
                'Caption',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_600'],
                alignment=TA_CENTER,
                spaceAfter=8,
                spaceBefore=10,
                wordWrap='CJK'
            ),
            
            # Metadata styles
            'meta_label': ParagraphStyle(
                'MetaLabel',
                fontName=self.font_body_bold,
                fontSize=8,
                textColor=self.colors['gray_700'],
                leading=13,
                wordWrap='CJK',
                letterSpacing=0.3
            ),
            
            'meta_value': ParagraphStyle(
                'MetaValue',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_600'],
                leading=13,
                wordWrap='CJK'
            ),
            
            # Disclaimer
            'disclaimer': ParagraphStyle(
                'Disclaimer',
                fontName=self.font_body,
                fontSize=8,
                textColor=self.colors['gray_600'],
                leading=13,
                leftIndent=0,
                rightIndent=0,
                spaceAfter=10,
                spaceBefore=10,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Clinical styles
            'clinical_summary': ParagraphStyle(
                'ClinicalSummary',
                fontName=self.font_body,
                fontSize=10,
                textColor=self.colors['gray_800'],
                leading=17,
                spaceAfter=8,
                alignment=TA_LEFT,
                wordWrap='CJK'
            ),
            
            # Stat styles
            'stat_value': ParagraphStyle(
                'StatValue',
                fontName=self.font_display,
                fontSize=18,
                textColor=self.colors['deepmind_primary'],
                alignment=TA_CENTER,
                spaceAfter=3,
                leading=22,
                wordWrap='CJK'
            ),
            
            'stat_label': ParagraphStyle(
                'StatLabel',
                fontName=self.font_body,
                fontSize=9,
                textColor=self.colors['gray_500'],
                alignment=TA_CENTER,
                leading=12,
                wordWrap='CJK'
            ),
        }

    def _create_header_canvas(self, canvas, doc):
        """Minimal, modern header in DeepMind style"""
        canvas.saveState()
        
        page_width = doc.pagesize[0]
        header_y = doc.pagesize[1] - 60
        
        # Thin accent line at top
        canvas.setStrokeColor(self.colors['deepmind_blue'])
        canvas.setLineWidth(2)
        canvas.line(0, doc.pagesize[1] - 1, page_width, doc.pagesize[1] - 1)
        
        # Logo/Brand mark
        logo_x = 40
        logo_y = header_y
        
        # Modern circular logo
        canvas.setFillColor(self.colors['deepmind_blue'])
        canvas.circle(logo_x + 12, logo_y + 12, 12, fill=1, stroke=0)
        canvas.setFillColor(self.colors['white'])
        canvas.setFont(self.font_display, 11)
        canvas.drawCentredString(logo_x + 12, logo_y + 7, "AI")
        
        # Brand text - clean and minimal
        canvas.setFont(self.font_display, 13)
        canvas.setFillColor(self.colors['deepmind_primary'])
        canvas.drawString(logo_x + 35, logo_y + 14, "DEEPMIND")
        
        canvas.setFont(self.font_body, 9)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(logo_x + 35, logo_y + 2, "Medical Intelligence")
        
        # Subtle separator
        canvas.setStrokeColor(self.colors['gray_200'])
        canvas.setLineWidth(0.5)
        canvas.line(40, header_y - 10, page_width - 40, header_y - 10)
        
        canvas.restoreState()

    def _create_footer_canvas(self, canvas, doc):
        """Minimal footer with essential information"""
        canvas.saveState()
        
        footer_y = 35
        page_width = doc.pagesize[0]
        
        # Top line
        canvas.setStrokeColor(self.colors['gray_200'])
        canvas.setLineWidth(0.5)
        canvas.line(40, 60, page_width - 40, 60)
        
        # Footer content
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(40, footer_y + 10, "CONFIDENTIAL MEDICAL ANALYSIS")
        canvas.drawString(40, footer_y - 2, "For licensed healthcare professionals only")
        
        # Page number - modern styling
        canvas.setFont(self.font_body_bold, 9)
        canvas.setFillColor(self.colors['deepmind_blue'])
        canvas.drawRightString(page_width - 40, footer_y + 4, f"{doc.page}")
        
        canvas.restoreState()

    def _create_deepmind_summary_card(self, findings):
        """Modern summary card with clean design"""
        is_positive = "positive" in findings.impression.lower()
        
        if is_positive:
            status_text = "ABNORMALITIES DETECTED"
            status_icon = "●"
            status_color = self.colors['alert_red']
            bg_color = self.colors['white']
        else:
            status_text = "NO SIGNIFICANT ABNORMALITIES"
            status_icon = "●"
            status_color = self.colors['science_green']
            bg_color = self.colors['white']
        
        status_style = ParagraphStyle(
            'StatusHeader',
            fontName=self.font_display,
            fontSize=12,
            textColor=status_color,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=12,
            wordWrap='CJK',
            letterSpacing=0.5
        )
        
        content = [
            [Paragraph(f"{status_icon} {status_text}", status_style)],
            [Paragraph(f"<b>Clinical Impression:</b> {findings.impression.replace('_', ' ').title()}", self.styles['clinical_summary'])],
            [Paragraph(f"<b>Diagnostic Confidence:</b> {findings.confidence_level}", self.styles['clinical_summary'])],
        ]
        
        summary_table = Table(content, colWidths=[6.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_300']),
            ('LINEABOVE', (0, 0), (-1, 0), 2, status_color),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
            ('LEFTPADDING', (0, 0), (-1, -1), 24),
            ('RIGHTPADDING', (0, 0), (-1, -1), 24),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        return summary_table

    def _create_deepmind_finding_card(self, condition, status, confidence, description=""):
        """Minimal, data-focused finding card"""
        if status == "Detected":
            status_color = self.colors['alert_red']
            bg_accent = self.colors['alert_red_light']
        else:
            status_color = self.colors['science_green']
            bg_accent = self.colors['success_light']
        
        try:
            conf_value = float(confidence.rstrip('%'))
            conf_display = f"{int(conf_value)}%"
        except:
            conf_display = "—"
        
        condition_style = ParagraphStyle(
            'ConditionName',
            fontName=self.font_body_bold,
            fontSize=11,
            textColor=self.colors['gray_900'],
            leading=15,
            wordWrap='CJK'
        )
        
        status_style = ParagraphStyle(
            'StatusText',
            fontName=self.font_body,
            fontSize=9,
            textColor=status_color,
            leading=13,
            wordWrap='CJK'
        )
        
        card_data = [
            [Paragraph(condition, condition_style), "", Paragraph(conf_display, ParagraphStyle('Conf', fontName=self.font_display, fontSize=14, textColor=self.colors['gray_800'], alignment=TA_RIGHT))],
            [Paragraph(status.upper(), status_style), "", ""]
        ]
        
        card_table = Table(card_data, colWidths=[4*inch, 0.5*inch, 1*inch])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_300']),
            ('LINEABOVE', (0, 0), (-1, 0), 2, status_color),
            ('SPAN', (0, 1), (1, 1)),
            ('TOPPADDING', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return card_table

    def _create_deepmind_metadata(self, analysis_response):
        """Clean metadata presentation"""
        metadata = analysis_response.image_metadata
        source = metadata.get('filename') or str(metadata.get('url', 'Unknown'))
        if len(source) > 60:
            source = source[:57] + '...'
        
        try:
            timestamp = datetime.fromisoformat(analysis_response.timestamp.replace('Z', '+00:00'))
            time_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')
        except:
            time_str = analysis_response.timestamp
        
        dimensions = metadata.get('dimensions', ['N/A', 'N/A'])
        
        metadata_items = [
            ('Analysis ID', analysis_response.analysis_id),
            ('Timestamp', time_str),
            ('Image Source', source),
            ('Dimensions', f"{dimensions[0]} × {dimensions[1]} px"),
            ('Model', "Med-GEMMA-3.1"),
        ]
        
        styled_rows = []
        for label, value in metadata_items:
            row = [
                Paragraph(label, self.styles['meta_label']),
                Paragraph(str(value), self.styles['meta_value'])
            ]
            styled_rows.append(row)
        
        meta_table = Table(styled_rows, colWidths=[1.2*inch, None])
        meta_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, -2), 0.25, self.colors['gray_200']),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
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
        """Generate a DeepMind-level sophisticated medical report"""
        
        filename = f"deepmind_medical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=90, bottomMargin=75
        )
        
        story = []
        findings = analysis_response.findings
        
        # === COVER ===
        story.append(Spacer(1, 30))
        story.append(Paragraph("DIAGNOSTIC IMAGING REPORT", self.styles['hero_title']))
        story.append(Paragraph("Chest Radiography Analysis", self.styles['main_title']))
        story.append(Spacer(1, 6))
        
        story.append(Paragraph(
            "Multi-condition screening for Pneumothorax, Pneumonia, and Tuberculosis using advanced neural networks",
            self.styles['subtitle']
        ))
        
        story.append(Spacer(1, 40))
        
        # === EXECUTIVE SUMMARY ===
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['section_header']))
        story.append(Spacer(1, 12))
        story.append(self._create_deepmind_summary_card(findings))
        story.append(Spacer(1, 30))
        
        # === KEY METRICS - Modern Cards ===
        impression_clean = findings.impression.title().replace('_', ' ')
        
        metric_cards = [
            DeepMindMetricCard(impression_clean, "Clinical Impression", width=2.2*inch),
            DeepMindMetricCard(findings.confidence_level, "Peak Confidence", "Model Certainty", width=2.2*inch),
            DeepMindMetricCard("3", "Conditions", "Screened", width=2.2*inch)
        ]
        
        metrics_table = Table([metric_cards], colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        metrics_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 35))
        
        # === CONFIDENCE VISUALIZATION ===
        story.append(Paragraph("MODEL CONFIDENCE", self.styles['section_header']))
        story.append(Spacer(1, 15))
        
        try:
            conf_val = float(findings.confidence_level.rstrip('%'))
        except:
            conf_val = 0
        
        gauge = ConfidenceGauge(conf_val)
        story.append(gauge)
        story.append(Spacer(1, 30))
        
        # === STUDY METADATA ===
        story.append(Paragraph("STUDY INFORMATION", self.styles['section_header']))
        story.append(Spacer(1, 10))
        story.append(self._create_deepmind_metadata(analysis_response))
        story.append(Spacer(1, 30))
        
        # === AI ANALYSIS ===
        story.append(Paragraph("NEURAL NETWORK ANALYSIS", self.styles['section_header']))
        story.append(Spacer(1, 8))
        
        analysis_text = analysis_response.raw_response.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
        analysis_table = Table([[Paragraph(analysis_text, self.styles['body'])]], colWidths=[6.5*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_300']),
            ('PADDING', (0, 0), (-1, -1), 20),
        ]))
        
        story.append(analysis_table)
        story.append(PageBreak())
        
        # === IMAGING ANALYSIS ===
        story.append(Paragraph("IMAGING ANALYSIS", self.styles['section_header']))
        story.append(Spacer(1, 15))
        
        img_max_width = 3 * inch
        rl_original = self._pil_to_reportlab_image(original_image, max_width=img_max_width, enhance=False)
        rl_heatmap = self._pil_to_reportlab_image(ai_attention_heatmap_image, max_width=img_max_width, enhance=False)
        
        image_data = [
            [rl_original, rl_heatmap],
            [Paragraph("Original Image", self.styles['caption']), Paragraph("Neural Attention Heatmap", self.styles['caption'])]
        ]
        
        image_table = Table(image_data, colWidths=[img_max_width + 0.2*inch, img_max_width + 0.2*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ]))
        
        story.append(image_table)
        story.append(Spacer(1, 15))
        
        story.append(Paragraph(
            "<i>Attention heatmap visualizes model focus areas. Warmer colors indicate regions of higher diagnostic relevance.</i>",
            self.styles['technical']
        ))
        story.append(Spacer(1, 35))
        
        # === RISK ASSESSMENT ===
        story.append(Paragraph("CONDITION ASSESSMENT", self.styles['section_header']))
        story.append(Spacer(1, 15))
        
        # Parse conditions
        abnormality_pattern = re.compile(r"(\w+)\s+\(([\d.]+%)\)")
        all_conditions = ["Pneumothorax", "Pneumonia", "TB"]
        detected = {m.group(1): m.group(2) for m in abnormality_pattern.finditer(" ".join(findings.abnormalities))}
        
        conditions_for_matrix = [(c, detected.get(c, "0%")) for c in all_conditions]
        risk_matrix = RiskMatrix(conditions_for_matrix)
        story.append(risk_matrix)
        story.append(Spacer(1, 25))
        
        # Individual condition cards
        for condition in all_conditions:
            if condition in detected:
                status = "Detected"
                confidence = detected[condition]
            else:
                status = "Not Detected"
                confidence = "N/A"
            
            story.append(self._create_deepmind_finding_card(condition, status, confidence))
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 30))
        
        # === CLINICAL RECOMMENDATIONS ===
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['section_header']))
        story.append(Spacer(1, 12))
        
        recommendations = [
            [Paragraph("<b>IMPORTANT NOTICE</b>", ParagraphStyle('Notice', parent=self.styles['body'], fontName=self.font_body_bold, fontSize=10, textColor=self.colors['alert_red'], alignment=TA_CENTER, spaceAfter=10))],
            [Paragraph("<b>Physician Review Required:</b> This AI analysis is a clinical decision support tool. All findings must be independently verified by a qualified radiologist before clinical action.", self.styles['body'])],
            [Paragraph("<b>Clinical Correlation:</b> Interpret imaging findings with patient history, examination, and laboratory results.", self.styles['body'])],
            [Paragraph("<b>Model Limitations:</b> AI models may have limitations with subtle findings or atypical presentations. Clinical judgment supersedes algorithmic output.", self.styles['body'])],
        ]
        
        rec_table = Table(recommendations, colWidths=[6.5*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['alert_red_light']),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_300']),
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.colors['alert_red']),
            ('PADDING', (0, 0), (-1, -1), 16),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(rec_table)
        story.append(Spacer(1, 20))
        
        # === ATTESTATION ===
        attestation = (
            "<b>Report Generation:</b> This report was automatically generated by the DeepMind Medical Intelligence system. "
            "Analysis based on deep learning trained on large-scale medical datasets. "
            "System designed to assist, not replace, professional medical judgment."
        )
        story.append(Paragraph(attestation, self.styles['disclaimer']))
        
        doc.build(story, onFirstPage=self._draw_page_template, onLaterPages=self._draw_page_template)
        
        logger.info(f"Generated DeepMind-style report at: {filepath}")
        return filepath

    def _draw_page_template(self, canvas, doc):
        """Helper method to draw the header and footer on each page."""
        self._create_header_canvas(canvas, doc)
        self._create_footer_canvas(canvas, doc)