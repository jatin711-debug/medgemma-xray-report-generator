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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as ReportLabImage, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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

class DeepMindMetricCard(Flowable):
    """Clean, modern metric card in DeepMind style"""
    def __init__(self, value, label, sublabel="", width=140, height=100, icon=""):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.value = str(value).strip()
        self.label = label
        self.sublabel = sublabel
        self.icon = icon
    
    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Subtle background with slight gradient effect
        canvas.setFillColor(colors.HexColor("#FAFAFA"))
        canvas.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        
        # Subtle border
        canvas.setStrokeColor(colors.HexColor("#E0E0E0"))
        canvas.setLineWidth(0.5)
        canvas.roundRect(0, 0, self.width, self.height, 6, fill=0, stroke=1)
        
        # Top accent line - thinner and more elegant
        canvas.setStrokeColor(colors.HexColor("#1976D2"))
        canvas.setLineWidth(2)
        canvas.line(0, self.height - 2, self.width, self.height - 2)
        
        # Icon if provided
        y_start = self.height - 25
        if self.icon:
            canvas.setFont("Helvetica", 16)
            canvas.setFillColor(colors.HexColor("#1976D2"))
            canvas.drawCentredString(self.width/2, y_start, self.icon)
            y_start -= 20
        
        # Value - handle long text with better wrapping
        value_text = self.value.replace('\n', ' ')
        
        # Dynamic font sizing
        if len(value_text) > 25:
            font_size = 12
            max_chars_per_line = 16
        elif len(value_text) > 20:
            font_size = 14
            max_chars_per_line = 18
        elif len(value_text) > 12:
            font_size = 18
            max_chars_per_line = 20
        else:
            font_size = 26
            max_chars_per_line = 100  # No wrapping needed
        
        canvas.setFont("Helvetica-Bold", font_size)
        canvas.setFillColor(colors.HexColor("#1A237E"))
        
        # Intelligent text wrapping
        if len(value_text) > max_chars_per_line:
            words = value_text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                test_length = current_length + len(word) + (1 if current_line else 0)
                if test_length > max_chars_per_line and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length = test_length
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw multiple lines centered
            line_height = font_size + 3
            total_height = len(lines) * line_height
            start_y = y_start - (total_height / 2) + line_height
            
            for i, line in enumerate(lines):
                canvas.drawCentredString(self.width/2, start_y - (i * line_height), line)
        else:
            canvas.drawCentredString(self.width/2, y_start - 10, value_text)
        
        # Label - uppercase and tracked
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#757575"))
        
        # Letter spacing simulation for label
        label_text = self.label.upper()
        canvas.drawCentredString(self.width/2, 18, label_text)
        
        # Sublabel
        if self.sublabel:
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.HexColor("#9E9E9E"))
            canvas.drawCentredString(self.width/2, 8, self.sublabel)
        
        canvas.restoreState()

class ConfidenceGauge(Flowable):
    """Enhanced DeepMind-style confidence gauge with better visuals"""
    def __init__(self, confidence_value, width=240, height=140):
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
        cy = 50
        radius = 45
        
        # Outer glow effect - multiple layers
        for i in range(3):
            glow_radius = radius + 12 - (i * 4)
            alpha = 0.03 + (i * 0.01)
            canvas.setStrokeColor(colors.Color(0.5, 0.7, 1.0, alpha=alpha))
            canvas.setLineWidth(8 - (i * 2))
            canvas.setLineCap(1)
            
            path = canvas.beginPath()
            for angle in range(180, 360, 2):
                rad = math.radians(angle)
                x = cx + glow_radius * math.cos(rad)
                y = cy + glow_radius * math.sin(rad)
                if angle == 180:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            canvas.drawPath(path, stroke=1, fill=0)
        
        # Background arc - clean and minimal
        canvas.setStrokeColor(colors.HexColor("#F0F0F0"))
        canvas.setLineWidth(12)
        canvas.setLineCap(1)
        
        path = canvas.beginPath()
        for angle in range(180, 360, 1):
            rad = math.radians(angle)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            if angle == 180:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        canvas.drawPath(path, stroke=1, fill=0)
        
        # Confidence arc with smooth gradient-like effect
        end_angle = 180 + (self.value / 100.0) * 180
        
        # Color based on confidence level
        if self.value >= 90:
            arc_color = colors.HexColor("#4CAF50")
            label_color = colors.HexColor("#2E7D32")
            desc = "Very High"
            desc_bg = colors.HexColor("#E8F5E9")
        elif self.value >= 75:
            arc_color = colors.HexColor("#2196F3")
            label_color = colors.HexColor("#1565C0")
            desc = "High"
            desc_bg = colors.HexColor("#E3F2FD")
        elif self.value >= 60:
            arc_color = colors.HexColor("#FFC107")
            label_color = colors.HexColor("#F57C00")
            desc = "Moderate"
            desc_bg = colors.HexColor("#FFF8E1")
        elif self.value >= 40:
            arc_color = colors.HexColor("#FF9800")
            label_color = colors.HexColor("#E65100")
            desc = "Fair"
            desc_bg = colors.HexColor("#FFE0B2")
        else:
            arc_color = colors.HexColor("#F44336")
            label_color = colors.HexColor("#C62828")
            desc = "Low"
            desc_bg = colors.HexColor("#FFEBEE")
        
        # Draw confidence arc with varying width for 3D effect
        for offset in range(3):
            alpha = 1.0 - (offset * 0.15)
            canvas.setStrokeColor(colors.Color(arc_color.red, arc_color.green, arc_color.blue, alpha=alpha))
            canvas.setLineWidth(12 - offset)
            
            path2 = canvas.beginPath()
            for angle_deg in range(180, int(end_angle), 1):
                rad = math.radians(angle_deg)
                x = cx + radius * math.cos(rad)
                y = cy + radius * math.sin(rad)
                if angle_deg == 180:
                    path2.moveTo(x, y)
                else:
                    path2.lineTo(x, y)
            canvas.drawPath(path2, stroke=1, fill=0)
        
        # Inner circle for clean center
        canvas.setFillColor(colors.white)
        canvas.circle(cx, cy, radius - 20, fill=1, stroke=0)
        
        # Subtle inner shadow
        canvas.setStrokeColor(colors.HexColor("#E0E0E0"))
        canvas.setLineWidth(0.5)
        canvas.circle(cx, cy, radius - 20, fill=0, stroke=1)
        
        # Center value - large and prominent
        canvas.setFillColor(colors.HexColor("#1A237E"))
        canvas.setFont("Helvetica-Bold", 32)
        canvas.drawCentredString(cx, cy - 10, f"{int(self.value)}%")
        
        # "CONFIDENCE" label below value
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#9E9E9E"))
        canvas.drawCentredString(cx, cy - 30, "CONFIDENCE")
        
        # Confidence level badge at bottom
        badge_y = 8
        badge_width = 80
        badge_height = 24
        badge_x = cx - (badge_width / 2)
        
        # Badge background with rounded corners
        canvas.setFillColor(desc_bg)
        canvas.roundRect(badge_x, badge_y, badge_width, badge_height, 12, fill=1, stroke=0)
        
        # Badge border
        canvas.setStrokeColor(label_color)
        canvas.setLineWidth(1.5)
        canvas.roundRect(badge_x, badge_y, badge_width, badge_height, 12, fill=0, stroke=1)
        
        # Badge text
        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(label_color)
        canvas.drawCentredString(cx, badge_y + 8, desc.upper())
        
        # Tick marks for scale
        canvas.setStrokeColor(colors.HexColor("#BDBDBD"))
        canvas.setLineWidth(1)
        
        for pct in [0, 25, 50, 75, 100]:
            angle = 180 + (pct / 100.0) * 180
            rad = math.radians(angle)
            
            # Outer tick
            x1 = cx + (radius + 2) * math.cos(rad)
            y1 = cy + (radius + 2) * math.sin(rad)
            x2 = cx + (radius + 8) * math.cos(rad)
            y2 = cy + (radius + 8) * math.sin(rad)
            
            canvas.line(x1, y1, x2, y2)
            
            # Tick label
            if pct in [0, 50, 100]:
                canvas.setFont("Helvetica", 7)
                canvas.setFillColor(colors.HexColor("#9E9E9E"))
                label_x = cx + (radius + 16) * math.cos(rad)
                label_y = cy + (radius + 16) * math.sin(rad) - 3
                canvas.drawCentredString(label_x, label_y, f"{pct}")
        
        canvas.restoreState()

class RiskMatrix(Flowable):
    """Enhanced DeepMind-style risk assessment matrix"""
    def __init__(self, conditions_data, width=340, height=155):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.conditions = conditions_data
    
    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Title with icon
        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(colors.HexColor("#424242"))
        canvas.drawString(0, self.height - 20, "📊 CONDITION RISK ASSESSMENT")
        
        bar_height = 26
        bar_spacing = 11
        start_y = self.height - 55
        label_width = 95
        bar_start = label_width + 12
        bar_max_width = self.width - bar_start - 45
        
        for i, (condition, confidence) in enumerate(self.conditions):
            y_pos = start_y - (i * (bar_height + bar_spacing))
            
            try:
                conf_val = float(str(confidence).rstrip('%'))
            except:
                conf_val = 0
            
            # Card background with subtle shadow
            shadow_offset = 2
            canvas.setFillColor(colors.Color(0, 0, 0, alpha=0.03))
            canvas.roundRect(0 + shadow_offset, y_pos - shadow_offset, self.width, bar_height, 4, fill=1, stroke=0)
            
            # Card background
            canvas.setFillColor(colors.white)
            canvas.roundRect(0, y_pos, self.width, bar_height, 4, fill=1, stroke=0)
            
            # Subtle border
            canvas.setStrokeColor(colors.HexColor("#E8E8E8"))
            canvas.setLineWidth(0.5)
            canvas.roundRect(0, y_pos, self.width, bar_height, 4, fill=0, stroke=1)
            
            # Background progress bar
            canvas.setFillColor(colors.HexColor("#F8F8F8"))
            canvas.roundRect(bar_start, y_pos + 3, bar_max_width, bar_height - 6, 3, fill=1, stroke=0)
            
            # Confidence bar with gradient effect
            bar_width = (conf_val / 100.0) * bar_max_width
            if bar_width > 0:
                if conf_val >= 70:
                    bar_color = colors.HexColor("#EF5350")
                    bar_light = colors.HexColor("#FF7875")
                    text_color = colors.white
                elif conf_val >= 40:
                    bar_color = colors.HexColor("#FF9800")
                    bar_light = colors.HexColor("#FFB74D")
                    text_color = colors.white
                else:
                    bar_color = colors.HexColor("#66BB6A")
                    bar_light = colors.HexColor("#81C784")
                    text_color = colors.white
                
                # Gradient effect with multiple layers
                for layer in range(3):
                    alpha = 1.0 - (layer * 0.15)
                    layer_color = colors.Color(bar_color.red, bar_color.green, bar_color.blue, alpha=alpha)
                    canvas.setFillColor(layer_color)
                    layer_height = (bar_height - 6 - (layer * 2))
                    canvas.roundRect(bar_start, y_pos + 3 + layer, bar_width, layer_height, 3, fill=1, stroke=0)
                
                # Percentage inside bar if space available
                if bar_width > 40:
                    canvas.setFont("Helvetica-Bold", 11)
                    canvas.setFillColor(text_color)
                    canvas.drawString(bar_start + 10, y_pos + 9, f"{int(conf_val)}%")
            
            # Condition label with icon
            condition_icons = {
                "Pneumothorax": "🫁",
                "Pneumonia": "🦠",
                "TB": "⚕️"
            }
            icon = condition_icons.get(condition, "•")
            
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.HexColor("#757575"))
            canvas.drawString(8, y_pos + 13, icon)
            
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.HexColor("#424242"))
            canvas.drawString(22, y_pos + 9, condition)
            
            # Percentage on right if not in bar
            if bar_width <= 40:
                canvas.setFont("Helvetica-Bold", 10)
                canvas.setFillColor(colors.HexColor("#757575"))
                canvas.drawRightString(self.width - 8, y_pos + 9, f"{int(conf_val)}%")
        
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
        """Minimal, modern header with SentryAI branding"""
        canvas.saveState()
        
        page_width = doc.pagesize[0]
        header_y = doc.pagesize[1] - 55
        
        # Very subtle top line
        canvas.setStrokeColor(self.colors['deepmind_blue'])
        canvas.setLineWidth(1.5)
        canvas.line(0, doc.pagesize[1] - 1, page_width, doc.pagesize[1] - 1)
        
        # Logo/Brand mark
        logo_x = 40
        logo_y = header_y
        
        # More refined shield logo
        canvas.setFillColor(self.colors['deepmind_blue'])
        shield_path = canvas.beginPath()
        shield_x = logo_x + 13
        shield_y = logo_y + 13
        shield_width = 11
        shield_height = 15
        
        # Shield with smoother curves
        shield_path.moveTo(shield_x - shield_width, shield_y + shield_height/2)
        shield_path.lineTo(shield_x - shield_width, shield_y - shield_height/3)
        shield_path.curveTo(
            shield_x - shield_width, shield_y - shield_height/2,
            shield_x - shield_width/2, shield_y - shield_height,
            shield_x, shield_y - shield_height
        )
        shield_path.curveTo(
            shield_x + shield_width/2, shield_y - shield_height,
            shield_x + shield_width, shield_y - shield_height/2,
            shield_x + shield_width, shield_y - shield_height/3
        )
        shield_path.lineTo(shield_x + shield_width, shield_y + shield_height/2)
        shield_path.curveTo(
            shield_x + shield_width, shield_y + shield_height*0.7,
            shield_x, shield_y + shield_height,
            shield_x, shield_y + shield_height
        )
        shield_path.curveTo(
            shield_x, shield_y + shield_height,
            shield_x - shield_width, shield_y + shield_height*0.7,
            shield_x - shield_width, shield_y + shield_height/2
        )
        shield_path.close()
        
        canvas.drawPath(shield_path, fill=1, stroke=0)
        
        # "AI" text with better positioning
        canvas.setFillColor(self.colors['white'])
        canvas.setFont(self.font_display, 10)
        canvas.drawCentredString(shield_x, shield_y - 4, "AI")
        
        # Brand text with better kerning
        canvas.setFont(self.font_display, 14)
        canvas.setFillColor(self.colors['deepmind_primary'])
        canvas.drawString(logo_x + 40, logo_y + 16, "SENTRYAI")
        
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(logo_x + 40, logo_y + 4, "Medical Intelligence Platform")
        
        # Minimal separator
        canvas.setStrokeColor(self.colors['gray_200'])
        canvas.setLineWidth(0.5)
        canvas.line(40, header_y - 8, page_width - 40, header_y - 8)
        
        canvas.restoreState()

    def _create_footer_canvas(self, canvas, doc):
        """Minimal footer with essential information"""
        canvas.saveState()
        
        footer_y = 32
        page_width = doc.pagesize[0]
        
        # Top line
        canvas.setStrokeColor(self.colors['gray_200'])
        canvas.setLineWidth(0.5)
        canvas.line(40, 58, page_width - 40, 58)
        
        # Footer content - more spaced
        canvas.setFont(self.font_body, 7)
        canvas.setFillColor(self.colors['gray_500'])
        canvas.drawString(40, footer_y + 12, "CONFIDENTIAL MEDICAL ANALYSIS")
        
        canvas.setFont(self.font_body, 7)
        canvas.setFillColor(self.colors['gray_400'])
        canvas.drawString(40, footer_y + 2, "For licensed healthcare professionals only • HIPAA Compliant")
        
        # Page number with circular background
        page_text = f"{doc.page}"
        canvas.setFillColor(self.colors['deepmind_ultra_light'])
        canvas.circle(page_width - 50, footer_y + 7, 12, fill=1, stroke=0)
        
        canvas.setFont(self.font_body_bold, 10)
        canvas.setFillColor(self.colors['deepmind_blue'])
        canvas.drawCentredString(page_width - 50, footer_y + 3, page_text)
        
        canvas.restoreState()

    def _create_deepmind_summary_card(self, findings):
        """Modern summary card with clean design"""
        is_positive = "positive" in findings.impression.lower()
        
        if is_positive:
            status_text = "ABNORMALITIES DETECTED"
            status_icon = "⚠"
            status_color = self.colors['alert_red']
            bg_color = self.colors['white']
            border_color = self.colors['alert_red']
        else:
            status_text = "NO SIGNIFICANT ABNORMALITIES"
            status_icon = "✓"
            status_color = self.colors['science_green']
            bg_color = self.colors['white']
            border_color = self.colors['science_green']
        
        status_style = ParagraphStyle(
            'StatusHeader',
            fontName=self.font_display,
            fontSize=13,
            textColor=status_color,
            leading=18,
            alignment=TA_LEFT,
            spaceAfter=14,
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
            ('BOX', (0, 0), (-1, -1), 1, self.colors['gray_200']),
            ('LINEABOVE', (0, 0), (-1, 0), 3, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 22),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 22),
            ('LEFTPADDING', (0, 0), (-1, -1), 28),
            ('RIGHTPADDING', (0, 0), (-1, -1), 28),
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
        
        filename = f"sentryai_medical_report_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=85, bottomMargin=70
        )
        
        story = []
        findings = analysis_response.findings
        
        # === COVER ===
        story.append(Spacer(1, 35))
        story.append(Paragraph("DIAGNOSTIC IMAGING REPORT", self.styles['hero_title']))
        story.append(Paragraph("Chest Radiography Analysis", self.styles['main_title']))
        story.append(Spacer(1, 8))
        
        story.append(Paragraph(
            "Multi-condition screening for Pneumothorax, Pneumonia, and Tuberculosis using advanced neural networks",
            self.styles['subtitle']
        ))
        
        story.append(Spacer(1, 45))
        
        # === EXECUTIVE SUMMARY ===
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['section_header']))
        story.append(Spacer(1, 14))
        story.append(self._create_deepmind_summary_card(findings))
        story.append(Spacer(1, 35))
        
        # === KEY METRICS - Modern Cards ===
        impression_clean = findings.impression.title().replace('_', ' ')
        
        metric_cards = [
            DeepMindMetricCard(impression_clean, "Clinical Impression", width=2.1*inch, height=100, icon="🔍"),
            DeepMindMetricCard(findings.confidence_level, "Peak Confidence", "Model Certainty", width=2.1*inch, height=100, icon="🎯"),
            DeepMindMetricCard("3", "Conditions", "Screened", width=2.1*inch, height=100, icon="📊")
        ]
        
        metrics_table = Table([metric_cards], colWidths=[2.1*inch, 2.1*inch, 2.1*inch])
        metrics_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 40))
        
        # === CONFIDENCE VISUALIZATION ===
        story.append(Paragraph("MODEL CONFIDENCE ASSESSMENT", self.styles['section_header']))
        story.append(Spacer(1, 18))
        
        try:
            conf_val = float(findings.confidence_level.rstrip('%'))
        except:
            conf_val = 0
        
        # Center the gauge
        gauge = ConfidenceGauge(conf_val)
        gauge_table = Table([[gauge]], colWidths=[3.5*inch])
        gauge_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ]))
        story.append(gauge_table)
        story.append(Spacer(1, 20))
        
        # Add interpretation text
        if conf_val >= 90:
            interp = "The model demonstrates <b>exceptional confidence</b> in its analysis, indicating strong pattern recognition across all evaluated conditions."
        elif conf_val >= 75:
            interp = "The model shows <b>high confidence</b> in its diagnostic assessment, with clear identification of relevant features."
        elif conf_val >= 60:
            interp = "The model exhibits <b>moderate confidence</b>. Consider additional clinical correlation for comprehensive evaluation."
        else:
            interp = "The model shows <b>limited confidence</b>. Further diagnostic imaging or clinical assessment is strongly recommended."
        
        interp_para = Paragraph(interp, self.styles['technical'])
        interp_table = Table([[interp_para]], colWidths=[5*inch])
        interp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('BOX', (0, 0), (-1, -1), 0.5, self.colors['gray_300']),
            ('PADDING', (0, 0), (-1, -1), 14),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(interp_table)
        story.append(Spacer(1, 35))
        
        # === STUDY METADATA ===
        story.append(Paragraph("STUDY INFORMATION", self.styles['section_header']))
        story.append(Spacer(1, 12))
        story.append(self._create_deepmind_metadata(analysis_response))
        story.append(Spacer(1, 35))
        
        # === AI ANALYSIS ===
        story.append(Paragraph("NEURAL NETWORK ANALYSIS", self.styles['section_header']))
        story.append(Spacer(1, 10))
        
        analysis_text = analysis_response.raw_response.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
        analysis_table = Table([[Paragraph(analysis_text, self.styles['body'])]], colWidths=[6.5*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 0.5, self.colors['gray_200']),
            ('PADDING', (0, 0), (-1, -1), 22),
        ]))
        
        story.append(analysis_table)
        story.append(PageBreak())
        
        # === IMAGING ANALYSIS ===
        story.append(Paragraph("IMAGING ANALYSIS", self.styles['section_header']))
        story.append(Spacer(1, 16))
        
        img_max_width = 3 * inch
        rl_original = self._pil_to_reportlab_image(original_image, max_width=img_max_width, enhance=False)
        rl_heatmap = self._pil_to_reportlab_image(ai_attention_heatmap_image, max_width=img_max_width, enhance=False)
        
        image_data = [
            [rl_original, rl_heatmap],
            [Paragraph("Original Image", self.styles['caption']), Paragraph("Neural Attention Heatmap", self.styles['caption'])]
        ]
        
        image_table = Table(image_data, colWidths=[img_max_width + 0.25*inch, img_max_width + 0.25*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 14),
        ]))
        
        story.append(image_table)
        story.append(Spacer(1, 18))
        
        story.append(Paragraph(
            "<i>Attention heatmap visualizes model focus areas during analysis. Warmer colors (red/yellow) indicate regions of higher diagnostic relevance and algorithmic attention.</i>",
            self.styles['technical']
        ))
        story.append(Spacer(1, 40))
        
        # === RISK ASSESSMENT ===
        story.append(Paragraph("CONDITION ASSESSMENT", self.styles['section_header']))
        story.append(Spacer(1, 16))
        
        # Parse conditions
        abnormality_pattern = re.compile(r"(\w+)\s+\(([\d.]+%)\)")
        all_conditions = ["Pneumothorax", "Pneumonia", "TB"]
        detected = {m.group(1): m.group(2) for m in abnormality_pattern.finditer(" ".join(findings.abnormalities))}
        
        conditions_for_matrix = [(c, detected.get(c, "0%")) for c in all_conditions]
        risk_matrix = RiskMatrix(conditions_for_matrix)
        story.append(risk_matrix)
        story.append(Spacer(1, 28))
        
        # Individual condition cards
        for condition in all_conditions:
            if condition in detected:
                status = "Detected"
                confidence = detected[condition]
            else:
                status = "Not Detected"
                confidence = "N/A"
            
            story.append(self._create_deepmind_finding_card(condition, status, confidence))
            story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 35))
        
        # === CLINICAL RECOMMENDATIONS ===
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['section_header']))
        story.append(Spacer(1, 14))
        
        recommendations = [
            [Paragraph("<b>⚠ IMPORTANT NOTICE</b>", ParagraphStyle('Notice', parent=self.styles['body'], fontName=self.font_body_bold, fontSize=11, textColor=self.colors['alert_red'], alignment=TA_CENTER, spaceAfter=12))],
            [Paragraph("<b>Physician Review Required:</b> This AI analysis serves as a clinical decision support tool. All findings must be independently verified by a qualified radiologist or physician before any clinical action is taken.", self.styles['body'])],
            [Paragraph("<b>Clinical Correlation:</b> Imaging findings should always be interpreted in conjunction with patient clinical history, physical examination findings, and laboratory results.", self.styles['body'])],
            [Paragraph("<b>Model Limitations:</b> AI models may have limitations in detecting subtle findings, rare conditions, or atypical presentations. Clinical judgment must always supersede algorithmic output.", self.styles['body'])],
        ]
        
        rec_table = Table(recommendations, colWidths=[6.5*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['alert_red_light']),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 0.5, self.colors['gray_200']),
            ('LINEABOVE', (0, 0), (-1, 0), 2, self.colors['alert_red']),
            ('PADDING', (0, 0), (-1, -1), 18),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(rec_table)
        story.append(Spacer(1, 22))
        
        # === ATTESTATION ===
        attestation = (
            "<b>Report Generation:</b> This report was automatically generated by the SentryAI Medical Intelligence Platform. "
            "Analysis is based on deep learning algorithms trained on large-scale medical imaging datasets. "
            "This system is designed to assist, not replace, professional medical judgment. "
            "Report generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        story.append(Paragraph(attestation, self.styles['disclaimer']))
        
        doc.build(story, onFirstPage=self._draw_page_template, onLaterPages=self._draw_page_template)
        
        logger.info(f"Generated SentryAI-style report at: {filepath}")
        return filepath

    def _draw_page_template(self, canvas, doc):
        """Helper method to draw the header and footer on each page."""
        self._create_header_canvas(canvas, doc)
        self._create_footer_canvas(canvas, doc)