import os
from xml.sax.saxutils import escape
import io
import logging
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
from src.reports.storage import atomic_report, report_path

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
            )
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
        canvas.drawString(logo_x + 60, logo_y + 25, "EXPERIMENTAL RESEARCH PROTOTYPE")
        
        canvas.setFont(self.font_body, 10)
        canvas.setFillColor(self.colors['gray_600'])
        canvas.drawString(logo_x + 60, logo_y + 8, "Experimental model output • Research use only")
        
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
        canvas.drawString(40, footer_y + 20, "Experimental research prototype")
        canvas.drawString(40, footer_y + 8, "Research use only • Not for clinical use")
        
        # Page number with style
        canvas.setFont(self.font_body_bold, 10)
        canvas.setFillColor(self.colors['primary'])
        canvas.drawRightString(page_width - 40, footer_y + 14, f"Page {doc.page}")
        
        # Security indicator
        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.colors['gray_400'])
        canvas.drawRightString(page_width - 40, footer_y + 2, "Not validated for patient care")
        
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
            [Paragraph(escape(str(value).replace('\n', ' ').strip()), value_style)],
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
            ('🎯', 'Model Version', "MedGemma (configured model)")
        ]
        
        styled_rows = []
        for icon, label, value in metadata_items:
            row = [
                Paragraph(icon, ParagraphStyle('Icon', fontSize=12, alignment=TA_CENTER)),
                Paragraph(label, self.styles['meta_label']),
                Paragraph(escape(str(value)), self.styles['meta_value'])
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
                            original_image: Image.Image) -> str:
        """Build a research PDF and publish it only after rendering succeeds."""
        destination = report_path(self.reports_dir, analysis_response.analysis_id)
        with atomic_report(destination) as temporary:
            self._build_pdf_report(analysis_response, original_image, temporary)
        logger.info("Successfully generated experimental PDF report")
        return str(destination)

    def _build_pdf_report(self, analysis_response, original_image: Image.Image,
                          filepath: str) -> None:
        """Render the document to a private temporary path before publication."""
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=100, bottomMargin=80
        )
        
        story = []
        findings = analysis_response.findings
        
        # === COVER PAGE ===
        story.append(Spacer(1, 25))
        story.append(Paragraph("PNEUMOTHORAX", self.styles['hero_title']))
        story.append(Paragraph("EXPERIMENTAL RESEARCH REPORT", self.styles['main_title']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(
            "Experimental model output for research only. Not for clinical use.",
            self.styles['subtitle']
        ))
        
        story.append(Spacer(1, 35))
        
        impression_icon = "📊"
        confidence_icon = "🎯"
        
        impression_color = 'warning'
        confidence_color = 'success' if 'HIGH' in findings.confidence_level.upper() else 'warning'
        
        stat_cards = [
            self._create_premium_stat_card(
                findings.impression.title().replace('_', ' '), 
                "Model Output",
                impression_color, 
                impression_icon
            ),
            self._create_premium_stat_card(
                findings.confidence_level.title(), 
                "Confidence",
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
        story.append(Spacer(1, 40))
        
        story.append(Paragraph("📋 Analysis Metadata", self.styles['section_header']))
        story.append(self._create_metadata_section(analysis_response))
        story.append(Spacer(1, 35))
        
        story.append(Paragraph("🧠 AI Analysis Output", self.styles['section_header']))
        analysis_text = escape(analysis_response.raw_response).replace('\n\n', '<br/><br/>').replace('\n', ' ')
        
        analysis_table = Table([[Paragraph(analysis_text, self.styles['body'])]], colWidths=[6.5*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['gray_50']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['primary_light']),
            ('PADDING', (0, 0), (-1, -1), 18),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [8, 8, 8, 8]),
        ]))
        
        story.append(analysis_table)
        story.append(Spacer(1, 35))
        
        story.append(Paragraph("⚖️ Medical & Legal Disclaimer", self.styles['section_header']))
        disclaimer_content = """<b>EXPERIMENTAL RESEARCH ONLY.</b> This output is not validated for diagnosis, clinical decisions, or patient care. Model findings may be incorrect or incomplete. Do not use real patient data without appropriate safeguards."""

        disclaimer_table = Table([[Paragraph(disclaimer_content, self.styles['disclaimer'])]], colWidths=[6.5*inch])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['warning_ultra_light']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['warning']),
            ('PADDING', (0, 0), (-1, -1), 16),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [8, 8, 8, 8]),
        ]))
        
        story.append(disclaimer_table)
        story.append(Spacer(1, 25))
        
        story.append(Paragraph("Input Image", self.styles['section_header']))
        story.append(self._pil_to_reportlab_image(original_image, max_width=3 * inch))
        story.append(Spacer(1, 25))

        # === DETAILED FINDINGS SECTION ===
        story.append(Paragraph("🔬 Detailed Technical Findings", self.styles['section_header']))
        
        detailed_findings_data = [
            [Paragraph("<b>Medical Finding</b>", self.styles['table_header']), 
             Paragraph("<b>Detection Status</b>", self.styles['table_header']), 
             Paragraph("<b>Model Output Details</b>", self.styles['table_header'])]
        ]
        
        finding_name = findings.impression
        status_p = Paragraph(escape(findings.impression.upper()), self.styles['finding_status'])
        details_text = "<br/>".join(escape(item) for item in findings.abnormalities)

        finding_p = Paragraph(escape(finding_name), self.styles['body'])
        details_p = Paragraph(details_text, self.styles['body'])
        detailed_findings_data.append([finding_p, status_p, details_p])

        findings_table = Table(detailed_findings_data, colWidths=[1.4*inch, 1.3*inch, 3.8*inch])
        findings_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['gray_300']),
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary_ultra_light']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['gray_50']]),
        ]))
        
        story.append(findings_table)
        story.append(Spacer(1, 25))

        # === CLINICAL RECOMMENDATIONS ===
        story.append(Paragraph("💡 Research Limitations", self.styles['section_header']))
        
        # Create a more compact recommendations section
        recommendations_data = [
            [Paragraph("<b>📋 Professional Review Required:</b> This prototype has not been validated for clinical decisions or patient care.", self.styles['body'])],
            [Paragraph("<b>Model Limitations:</b> The model may produce incorrect or indeterminate findings.", self.styles['body'])],
            [Paragraph("<b>Confidence:</b> No calibrated confidence score is available.", self.styles['body'])]
        ]
        
        recommendations_table = Table(recommendations_data, colWidths=[6.5*inch])
        recommendations_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['primary_ultra_light']),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['primary_light']),
            ('PADDING', (0, 0), (-1, -1), 14),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [6, 6, 6, 6]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(recommendations_table)

        doc.build(story, onFirstPage=self._draw_page_template, onLaterPages=self._draw_page_template)
        
    def _draw_page_template(self, canvas, doc):
        """Helper method to draw the header and footer on each page."""
        self._create_header_canvas(canvas, doc)
        self._create_footer_canvas(canvas, doc)
