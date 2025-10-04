import os
import io
import logging
from datetime import datetime
from PIL import Image, ImageFilter, ImageDraw

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as ReportLabImage, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#region: Custom Flowables for Data Visualization

class IconFlowable(Flowable):
    """Draws clean, geometric icons instead of emojis."""
    def __init__(self, icon_type='dot', size=10, color=colors.black):
        super().__init__()
        self.icon_type = icon_type
        self.size = size
        self.color = color

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(self.color)
        canvas.setStrokeColor(self.color)
        
        radius = self.size / 2.0
        if self.icon_type == 'check':
            canvas.setLineWidth(1.5)
            canvas.circle(radius, radius, radius, fill=0)
            canvas.polyline([self.size*0.25, self.size*0.5, self.size*0.45, self.size*0.7, self.size*0.75, self.size*0.3])
        elif self.icon_type == 'alert':
            canvas.setLineWidth(1.5)
            path = canvas.beginPath()
            path.moveTo(self.size/2, self.size)
            path.lineTo(0, 0)
            path.lineTo(self.size, 0)
            path.close()
            canvas.drawPath(path, stroke=1, fill=0)
            canvas.circle(self.size/2, self.size*0.3, 0.5) # Dot inside
        else: # Default to dot
            canvas.circle(radius, radius, radius, fill=1)
            
        canvas.restoreState()

class ConfidenceBar(Flowable):
    """A custom flowable to draw a confidence bar, making percentages visual."""
    def __init__(self, width, height, value, text, color_map):
        super().__init__()
        self.width = width
        self.height = height
        self.value = value  # Expects a value from 0 to 1
        self.text = text
        self.color = self._get_color(value, color_map)

    def _get_color(self, value, color_map):
        if value >= 0.8: return color_map['high']
        if value >= 0.5: return color_map['medium']
        return color_map['low']

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Draw background track
        canvas.setFillColor(colors.HexColor("#E5E7EB"))
        canvas.roundRect(0, 0, self.width, self.height, self.height/2, stroke=0, fill=1)
        
        # Draw filled bar
        canvas.setFillColor(self.color)
        fill_width = self.width * self.value
        canvas.roundRect(0, 0, fill_width, self.height, self.height/2, stroke=0, fill=1)
        
        # Draw text label inside the bar
        canvas.setFillColor(colors.white)
        canvas.setFont('Roboto-Medium', 7)
        # Position text to be visible
        text_x = fill_width - 25 if fill_width > 30 else 5
        canvas.drawString(text_x, self.height / 4, self.text)
        
        canvas.restoreState()

#endregion

class DeepMindStylePDFGenerator:
    """
    Generates a visually stunning medical report inspired by Google DeepMind's
    clean, modern, and data-centric design philosophy.
    """
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self._setup_palette()
        self._setup_fonts()
        self._setup_styles()

    def _setup_fonts(self):
        """Setup a clean, modern typography system using Roboto."""
        try:
            pdfmetrics.registerFont(TTFont('Roboto', './assets/Roboto-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Roboto-Medium', './assets/Roboto-Medium.ttf'))
            pdfmetrics.registerFont(TTFont('Roboto-Light', './assets/Roboto-Light.ttf'))
            self.font_display = 'Roboto-Light'
            self.font_body = 'Roboto'
            self.font_body_medium = 'Roboto-Medium'
            logger.info("Successfully loaded Roboto font family.")
        except Exception:
            logger.warning("Roboto fonts not found in 'assets/'. Using Helvetica as fallback.")
            self.font_display = 'Helvetica'
            self.font_body = 'Helvetica'
            self.font_body_medium = 'Helvetica-Bold'

    def _setup_palette(self):
        """A refined, modern color palette inspired by tech aesthetics."""
        self.palette = {
            'primary': colors.HexColor("#1A73E8"),         # Google Blue
            'primary_dark': colors.HexColor("#0D47A1"),
            'accent': colors.HexColor("#00A79D"),          # Teal
            'danger': colors.HexColor("#D93025"),          # Google Red
            'success': colors.HexColor("#1E8E3E"),         # Google Green
            'warning': colors.HexColor("#F9AB00"),         # Google Yellow
            'text_primary': colors.HexColor("#202124"),    # Almost Black
            'text_secondary': colors.HexColor("#5F6368"),  # Dark Gray
            'bg_light': colors.HexColor("#F8F9FA"),        # Light Gray Background
            'border': colors.HexColor("#DADCE0"),          # Light Border
            'white': colors.white,
        }

    def _setup_styles(self):
        """Create a consistent and readable set of paragraph styles."""
        self.styles = {
            'hero_title': ParagraphStyle(
                'HeroTitle', fontName=self.font_display, fontSize=32, 
                textColor=self.palette['text_primary'], leading=40, spaceAfter=8
            ),
            'main_title': ParagraphStyle(
                'MainTitle', fontName=self.font_body_medium, fontSize=18,
                textColor=self.palette['primary'], leading=24, spaceAfter=24
            ),
            'section_header': ParagraphStyle(
                'SectionHeader', fontName=self.font_body_medium, fontSize=14,
                textColor=self.palette['primary_dark'], leading=18, spaceBefore=18, spaceAfter=12
            ),
            'body': ParagraphStyle(
                'Body', fontName=self.font_body, fontSize=10,
                textColor=self.palette['text_secondary'], leading=16, alignment=TA_JUSTIFY
            ),
            'meta_label': ParagraphStyle(
                'MetaLabel', fontName=self.font_body_medium, fontSize=9, textColor=self.palette['text_primary']
            ),
            'meta_value': ParagraphStyle(
                'MetaValue', fontName=self.font_body, fontSize=9, textColor=self.palette['text_secondary']
            ),
            'caption': ParagraphStyle(
                'Caption', fontName=self.font_body, fontSize=9, textColor=self.palette['text_secondary'],
                alignment=TA_CENTER, spaceBefore=6
            ),
            'disclaimer': ParagraphStyle(
                'Disclaimer', fontName=self.font_body, fontSize=8, textColor=self.palette['text_secondary'],
                leading=12, alignment=TA_CENTER
            ),
            'finding_condition': ParagraphStyle(
                'FindingCondition', fontName=self.font_body_medium, fontSize=12, textColor=self.palette['text_primary']
            ),
            'finding_status_detected': ParagraphStyle(
                'FindingStatusDetected', fontName=self.font_body, fontSize=10, textColor=self.palette['danger']
            ),
            'finding_status_normal': ParagraphStyle(
                'FindingStatusNormal', fontName=self.font_body, fontSize=10, textColor=self.palette['success']
            ),
        }

    def _color_to_pil_rgb(self, color):
        """Convert reportlab Color object to PIL-compatible RGB tuple (0-255 range)."""
        if hasattr(color, 'rgb'):
            r, g, b = color.rgb()
            return (int(r * 255), int(g * 255), int(b * 255))
        elif hasattr(color, 'red'):
            return (int(color.red * 255), int(color.green * 255), int(color.blue * 255))
        else:
            # Fallback for hex colors or other formats
            return (128, 128, 128)  # Gray fallback

    def _on_page(self, canvas, doc):
        """Draws the header and footer on each page."""
        canvas.saveState()
        
        # -- Header --
        header_height = 0.75 * inch
        canvas.setFillColor(self.palette['bg_light'])
        canvas.rect(0, doc.height + doc.topMargin - header_height, doc.width, header_height, stroke=0, fill=1)
        canvas.setStrokeColor(self.palette['border'])
        canvas.line(0, doc.height + doc.topMargin - header_height, doc.width, doc.height + doc.topMargin - header_height)
        
        canvas.setFont(self.font_body_medium, 10)
        canvas.setFillColor(self.palette['text_primary'])
        canvas.drawString(doc.leftMargin, doc.height + 0.35 * inch, "AI Diagnostic Imaging Report")
        
        # -- Footer --
        footer_height = 0.75 * inch
        canvas.setFillColor(self.palette['bg_light'])
        canvas.rect(0, 0, doc.width, footer_height, stroke=0, fill=1)
        canvas.setStrokeColor(self.palette['border'])
        canvas.line(0, footer_height, doc.width, footer_height)

        canvas.setFont(self.font_body, 8)
        canvas.setFillColor(self.palette['text_secondary'])
        canvas.drawCentredString(doc.width/2, 0.4 * inch, "Med-GEMMA-4 Analysis Engine | Confidential | For Professional Use Only")
        canvas.drawRightString(doc.width - doc.rightMargin, 0.4 * inch, f"Page {doc.page}")

        canvas.restoreState()

    def _create_shadowed_image(self, pil_image, max_width, shadow_offset=5, bg_color="#F8F9FA"):
        """Adds a subtle border and shadow to an image for a premium feel."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Add border - convert reportlab color to PIL RGB tuple
        border_rgb = self._color_to_pil_rgb(self.palette['border'])
        bordered = Image.new('RGB', (pil_image.width + 2, pil_image.height + 2), border_rgb)
        bordered.paste(pil_image, (1, 1))

        # Create shadow
        shadow = Image.new('RGB', (bordered.width + shadow_offset, bordered.height + shadow_offset), bg_color)
        shadow_layer = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow_layer).rounded_rectangle(
            (shadow_offset, shadow_offset, bordered.width + shadow_offset, bordered.height + shadow_offset),
            radius=4, fill=(0, 0, 0, 60) # Black with alpha
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(3))
        shadow.paste(Image.alpha_composite(Image.new('RGBA', shadow.size, (0,0,0,0)), shadow_layer), (0,0))
        shadow.paste(bordered, (0, 0))

        # Convert to ReportLab Image
        img_buffer = io.BytesIO()
        shadow.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        aspect = shadow.height / float(shadow.width)
        return ReportLabImage(img_buffer, width=max_width, height=max_width * aspect)

    def _create_executive_summary(self, findings):
        """Creates a clean, modern executive summary card."""
        is_positive = "positive" in findings.impression.lower()
        status_color = self.palette['danger'] if is_positive else self.palette['success']
        status_text = "Abnormalities Detected" if is_positive else "No Significant Abnormalities Detected"

        icon = IconFlowable('alert', color=status_color) if is_positive else IconFlowable('check', color=status_color)
        
        summary_style = ParagraphStyle(
            'Summary', parent=self.styles['body'], 
            textColor=self.palette['text_primary'], fontSize=11, leading=18
        )

        # --- FIX IS HERE ---
        # Get the color object and reliably convert it to a hex string
        text_secondary_color_obj = self.palette['text_secondary']
        hex_color_str = f"#{int(text_secondary_color_obj.red*255):02x}{int(text_secondary_color_obj.green*255):02x}{int(text_secondary_color_obj.blue*255):02x}"
        
        summary_paragraph = Paragraph(
            f"<b>{status_text}</b><br/><font color='{hex_color_str}'>Impression: {findings.impression.replace('_', ' ').title()}</font>",
            summary_style
        )
        
        content = [
            [
                icon,
                summary_paragraph
            ]
        ]
        
        table = Table(content, colWidths=[0.3*inch, None])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('BACKGROUND', (0, 0), (-1, -1), self.palette['bg_light']),
            ('BOX', (0, 0), (-1, -1), 1, self.palette['border']),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [4, 4, 4, 4]),
            ('PADDING', (0, 0), (-1, -1), 16),
            ('LINEBEFORE', (0, 0), (0, -1), 4, status_color), # Key design element: side-border
        ]))
        return table
    def _create_diagnostic_finding_card(self, condition, status, confidence_str):
        """Creates an individual finding card with a confidence bar."""
        try:
            confidence_val = float(confidence_str.strip('%')) / 100.0
        except:
            confidence_val = 0.0

        if status == "Detected":
            status_style = self.styles['finding_status_detected']
            bar_color_map = {'high': self.palette['danger'], 'medium': self.palette['warning'], 'low': self.palette['text_secondary']}
        else:
            status_style = self.styles['finding_status_normal']
            bar_color_map = {'high': self.palette['success'], 'medium': self.palette['success'], 'low': self.palette['success']}
            confidence_val = 0 # No bar for "not detected"

        confidence_bar = ConfidenceBar(
            width=1.5 * inch, height=12, value=confidence_val, text=confidence_str, color_map=bar_color_map
        )

        content = [
            [
                Paragraph(condition, self.styles['finding_condition']),
                Paragraph(status, status_style),
                confidence_bar if confidence_str != "N/A" else Paragraph("—", self.styles['body'])
            ]
        ]
        
        table = Table(content, colWidths=[None, 1.0*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, self.palette['border']),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        return table

    def _create_metadata_table(self, analysis_response):
        """Creates a clean, two-column metadata table."""
        metadata = analysis_response.image_metadata
        try:
            ts = datetime.fromisoformat(analysis_response.timestamp.replace('Z', '+00:00'))
            time_str = ts.strftime('%B %d, %Y, %H:%M:%S UTC')
        except:
            time_str = analysis_response.timestamp

        items = {
            "Analysis ID": analysis_response.analysis_id,
            "Generated": time_str,
            "Image Source": metadata.get('filename', 'N/A'),
            "Model Version": "Med-GEMMA-4.0 (Build 2025.10)"
        }
        
        data = [[Paragraph(k, self.styles['meta_label']), Paragraph(v, self.styles['meta_value'])] for k,v in items.items()]
        
        table = Table(data, colWidths=[1.5*inch, None])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, self.palette['border']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        return table

    def generate_pdf_report(self,
                            analysis_response,
                            original_image: Image.Image,
                            ai_attention_heatmap_image: Image.Image) -> str:
        
        filename = f"MedReport_DeepMind_{analysis_response.analysis_id}.pdf"
        filepath = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath, pagesize=letter,
            rightMargin=inch, leftMargin=inch,
            topMargin=inch, bottomMargin=inch
        )
        
        story = []
        findings = analysis_response.findings
        
        # === COVER PAGE LAYOUT ===
        story.append(Paragraph("Diagnostic Imaging Report", self.styles['hero_title']))
        story.append(Paragraph("AI-Assisted Chest Radiograph Analysis", self.styles['main_title']))
        
        # --- Top section: Metadata and Summary side-by-side ---
        meta_table = self._create_metadata_table(analysis_response)
        summary_card = self._create_executive_summary(findings)
        
        top_section_table = Table([[meta_table, summary_card]], colWidths=[3.2*inch, None], spaceAfter=20)
        top_section_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        story.append(top_section_table)
        story.append(Spacer(1, 0.3 * inch))

        # === AI INTERPRETATION (RAW RESPONSE) ===
        story.append(Paragraph("Neural Network Interpretation", self.styles['section_header']))
        raw_text = analysis_response.raw_response.replace('\n', '<br/>')
        story.append(Paragraph(raw_text, self.styles['body']))
        story.append(Spacer(1, 0.3 * inch))
        story.append(PageBreak())

        # === IMAGING ANALYSIS ===
        story.append(Paragraph("Imaging & Neural Attention Analysis", self.styles['section_header']))
        
        img_width = 3 * inch
        rl_original = self._create_shadowed_image(original_image, img_width)
        rl_heatmap = self._create_shadowed_image(ai_attention_heatmap_image, img_width)
        
        image_table = Table(
            [[rl_original, rl_heatmap]], 
            colWidths=[img_width + 0.2*inch, img_width + 0.2*inch]
        )
        image_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        caption_table = Table(
            [[
                Paragraph("Original Radiograph", self.styles['caption']),
                Paragraph("Neural Attention Map", self.styles['caption'])
            ]],
            colWidths=[img_width + 0.2*inch, img_width + 0.2*inch]
        )
        
        story.append(KeepTogether([image_table, caption_table]))
        story.append(Spacer(1, 0.3 * inch))

        # === DETAILED FINDINGS ===
        story.append(Paragraph("Detailed Pathological Findings", self.styles['section_header']))
        
        all_conditions = ["Pneumothorax", "Pneumonia", "Tuberculosis"] # Use consistent naming
        detected_conditions = {ab.split(' ')[0]: ab.split('(')[1].strip(')') for ab in findings.abnormalities}

        finding_cards = []
        for condition in all_conditions:
            if condition in detected_conditions:
                status = "Detected"
                confidence = detected_conditions[condition]
            else:
                status = "Not Detected"
                confidence = "N/A"
            finding_cards.append(self._create_diagnostic_finding_card(condition, status, confidence))

        findings_table = Table(
            [[card] for card in finding_cards], 
            colWidths=[6.5 * inch]
        )
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), self.palette['bg_light']),
            ('BOX', (0, 0), (-1, -1), 1, self.palette['border']),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), [4, 4, 4, 4]),
            ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('LEFTPADDING', (0, 0), (-1, -1), 12), ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(findings_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # === DISCLAIMER ===
        disclaimer_text = (
            "<b>Important:</b> This AI-generated report is a clinical decision support tool and not a substitute for "
            "professional medical advice. All findings must be reviewed and verified by a qualified physician, "
            "correlating with the patient's full clinical context."
        )
        story.append(Paragraph(disclaimer_text, self.styles['disclaimer']))
        
        doc.build(story, onFirstPage=self._on_page, onLaterPages=self._on_page)
        
        logger.info(f"Successfully generated DeepMind-style PDF report: {filepath}")
        return filepath