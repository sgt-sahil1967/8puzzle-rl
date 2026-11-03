"""Small report-layout toolkit on top of ReportLab (chapters, numbered figures/tables, contents pages)."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as _canvas
from reportlab.platypus import (CondPageBreak, BaseDocTemplate, Flowable, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from PIL import Image as PILImage

SHORT = "8-Puzzle Solver – Q-Learning vs DQN vs A*"
HEADER = "Deep Learning Mini-Project 2026–27"
W, H = A4
LM = RM = 2.5 * cm
FW = W - LM - RM  # frame width

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9.5, leading=12, alignment=TA_LEFT, spaceAfter=0)
SMALLC = ParagraphStyle("smallc", parent=SMALL, alignment=TA_CENTER)
SMALLB = ParagraphStyle("smallb", parent=SMALL, fontName="Times-Bold")
SMALLBC = ParagraphStyle("smallbc", parent=SMALLB, alignment=TA_CENTER)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=10, leading=13, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)
CHLABEL = ParagraphStyle("chl", fontName="Times-Bold", fontSize=22, leading=28, spaceAfter=2)
CHTITLE = ParagraphStyle("cht", fontName="Times-Bold", fontSize=26, leading=32, spaceAfter=22)
SEC = ParagraphStyle("sec", fontName="Times-Bold", fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, keepWithNext=1)
SUB = ParagraphStyle("sub", fontName="Times-Bold", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=18, bulletIndent=4, spaceAfter=3)
QUOTE = ParagraphStyle("quote", parent=BODY, leftIndent=24, rightIndent=24, fontName="Times-Italic")
CODE = ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=10.6, leftIndent=6, backColor=colors.HexColor("#f4f4f4"), borderPadding=5)


class Heading(Paragraph):
    """Paragraph that remembers what it is, so contents pages can be built from it."""
    def __init__(self, text, style, kind, label=""):
        super().__init__(text, style)
        self.kind, self.label, self.plain = kind, label, text


class Caption(Paragraph):
    def __init__(self, text, kind, label):
        super().__init__("%s %s: %s" % ("Figure" if kind == "fig" else "Table", label, text), CAP)
        self.kind, self.label, self.plain = kind, label, text


class TocLine(Flowable):
    """One line of a contents page: text, dot leader, page number."""
    def __init__(self, label, text, page, indent=0, bold=False):
        super().__init__()
        self.label, self.page, self.indent, self.bold = label, page, indent, bold
        self.text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        self.height = 15.5 if bold else 14

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        c, font = self.canv, "Times-Bold" if self.bold else "Times-Roman"
        c.setFont(font, 11)
        left = self.indent
        if self.label:
            c.drawString(left, 2, self.label)
            left += max(26, c.stringWidth(self.label, font, 11) + 8)
        c.drawString(left, 2, self.text)
        end = left + c.stringWidth(self.text, font, 11) + 4
        pw = c.stringWidth(str(self.page), font, 11)
        c.setFont("Times-Roman", 11)
        x = end
        dw = c.stringWidth(" .", "Times-Roman", 11)
        while x + dw < self.width - pw - 6:
            c.drawString(x, 2, " .")
            x += dw
        c.setFont(font, 11)
        c.drawRightString(self.width, 2, str(self.page))


def roman(n):
    out = ""
    for v, s in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out, n = out + s, n - v
    return out


class ReportDoc(BaseDocTemplate):
    def __init__(self, path, state, prev):
        super().__init__(path, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=2.6 * cm, bottomMargin=2.3 * cm,
                         title="8-Puzzle Solver: Q-Learning vs DQN vs A* - Deep Learning Mini-Project Report",
                         author="Bhavsar, Patil, Dhokale, Shinde")
        self.state, self.prev = state, prev
        frame = Frame(LM, 2.3 * cm, FW, H - 2.6 * cm - 2.3 * cm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self.decorate)])

    def display_page(self, p):
        ms = self.prev.get("main_start")
        if ms is None or p < ms:
            return roman(p - 1) if p > 1 else ""
        return str(p - ms + 1)

    def decorate(self, c, doc):
        p = doc.page
        if p == 1:
            return
        c.saveState()
        c.setFont("Times-Roman", 10)
        if p not in self.prev.get("chapter_pages", ()):
            c.drawString(LM, H - 1.7 * cm, SHORT)
            c.drawRightString(W - RM, H - 1.7 * cm, HEADER)
            c.setLineWidth(0.5)
            c.line(LM, H - 1.85 * cm, W - RM, H - 1.85 * cm)
        c.drawCentredString(W / 2, 1.3 * cm, self.display_page(p))
        c.restoreState()

    def afterFlowable(self, f):
        s = self.state
        if isinstance(f, (Heading, Caption)):
            if f.kind == "chapter" and "main_start" not in s:
                s["main_start"] = self.page
            if f.kind in ("chapter", "unnumbered"):
                s.setdefault("chapter_pages", set()).add(self.page)
            s["entries"].append((f.kind, f.label, f.plain, self.display_page_now(self.page)))

    def display_page_now(self, p):
        ms = self.state.get("main_start")
        if ms is None or p < ms:
            return roman(p - 1) if p > 1 else ""
        return str(p - ms + 1)


def fit_image(path, width_cm=None, max_height_cm=None):
    w, h = PILImage.open(path).size
    width = (width_cm or 14) * cm
    height = width * h / w
    if max_height_cm and height > max_height_cm * cm:
        height = max_height_cm * cm
        width = height * w / h
    return Image(path, width=width, height=height)
