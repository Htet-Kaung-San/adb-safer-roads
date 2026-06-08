"""
Convert findings_summary_5page.md to a polished PDF using reportlab.
Usage: python scripts/generate_pdf_report.py
Output: report/findings_summary_5page.pdf
"""
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

ROOT = Path(__file__).parent.parent
MD   = ROOT / "report" / "findings_summary_5page.md"
PDF  = ROOT / "report" / "findings_summary_5page.pdf"

# ── Colours ────────────────────────────────────────────────────────────────
BLUE      = HexColor("#1E40AF")
BLUE_LIGHT= HexColor("#DBEAFE")
RED       = HexColor("#DC2626")
ORANGE    = HexColor("#EA580C")
GREEN     = HexColor("#16A34A")
GREY_BG   = HexColor("#F8FAFC")
GREY_RULE = HexColor("#CBD5E1")
TEXT      = HexColor("#1E293B")
MUTED     = HexColor("#64748B")

W, H = A4   # 595.28 x 841.89 pts

def make_styles():
    base = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "title":    S("title",    fontSize=18, textColor=BLUE,    fontName="Helvetica-Bold",
                       leading=24, spaceAfter=3),
        "subtitle": S("subtitle", fontSize=10.5, textColor=MUTED, fontName="Helvetica",
                       leading=15, spaceAfter=10),
        "h1":       S("h1",       fontSize=11.5, textColor=white,  fontName="Helvetica-Bold",
                       leading=16, spaceAfter=0, spaceBefore=12),
        "h2":       S("h2",       fontSize=10.5, textColor=BLUE,   fontName="Helvetica-Bold",
                       leading=14, spaceAfter=4, spaceBefore=10),
        "body":     S("body",     fontSize=10, textColor=TEXT,     fontName="Helvetica",
                       leading=15.5, spaceAfter=7, alignment=TA_JUSTIFY),
        "bullet":   S("bullet",   fontSize=10, textColor=TEXT,     fontName="Helvetica",
                       leading=14.5, spaceAfter=4, leftIndent=14, firstLineIndent=-10),
        "italic":   S("italic",   fontSize=8.8, textColor=MUTED,   fontName="Helvetica-Oblique",
                       leading=12, spaceAfter=12, alignment=TA_CENTER),
        "code":     S("code",     fontSize=8.2, textColor=TEXT,    fontName="Courier",
                       leading=12, spaceAfter=5, backColor=GREY_BG, leftIndent=10, rightIndent=10),
    }


def grade_color(grade):
    return {"A": GREEN, "B": HexColor("#CA8A04"), "C": ORANGE, "D": RED}.get(grade.strip("* "), BLUE)


def inline(text):
    """Convert **bold**, *italic*, `code` to ReportLab markup."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",       r'<font name="Courier" size="7.5">\1</font>', text)
    return text


def table_style(header_bg=BLUE, stripe=GREY_BG):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  6),
        ("TOPPADDING",    (0, 0), (-1, 0),  6),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, stripe]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_RULE),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ])


def parse_md(md_path, styles, avail_w):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0

    def add(item):
        story.append(item)

    def body(text):
        return Paragraph(inline(text), styles["body"])

    while i < len(lines):
        line = lines[i]

        # ── Title / subtitle ──────────────────────────────────────────────
        if line.startswith("# ") and not line.startswith("## "):
            add(Paragraph(line[2:], styles["title"]))
            i += 1
            if i < len(lines) and lines[i].startswith("## "):
                add(Paragraph(lines[i][3:], styles["subtitle"]))
                i += 1
            if i < len(lines) and lines[i].startswith("**Team:"):
                add(Paragraph(inline(lines[i].strip("*")), styles["subtitle"]))
                i += 1
            add(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=8))
            continue

        # ── Section headings (##) ─────────────────────────────────────────
        if line.startswith("## "):
            text = line[3:].strip()
            # Blue pill header
            t = Table([[Paragraph(text, styles["h1"])]], colWidths=[avail_w])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), BLUE),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ]))
            add(Spacer(1, 4))
            add(t)
            add(Spacer(1, 4))
            i += 1
            continue

        # ── Sub-headings (###) ────────────────────────────────────────────
        if line.startswith("### "):
            add(Paragraph(line[4:], styles["h2"]))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────
        if line.strip() == "---":
            add(HRFlowable(width="100%", thickness=0.5, color=GREY_RULE, spaceAfter=4, spaceBefore=4))
            i += 1
            continue

        # ── Tables ────────────────────────────────────────────────────────
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # filter separator row
            rows = [r for r in table_lines if not re.match(r"^\|\s*[-:]+", r)]
            if rows:
                data = []
                for ri, row in enumerate(rows):
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    style_name = "Helvetica-Bold" if ri == 0 else "Helvetica"
                    data.append([
                        Paragraph(
                            f"<b>{inline(c)}</b>" if ri == 0 else inline(c),
                            ParagraphStyle("tc", fontName=style_name, fontSize=9,
                                           textColor=white if ri == 0 else TEXT,
                                           leading=12, alignment=TA_LEFT)
                        ) for c in cells
                    ])
                ncols = len(data[0])
                col_w = avail_w / ncols
                t = Table(data, colWidths=[col_w] * ncols)
                t.setStyle(table_style())
                add(KeepTogether([t, Spacer(1, 4)]))
            continue

        # ── Bullet points ─────────────────────────────────────────────────
        if line.startswith("- ") or line.startswith("* "):
            add(Paragraph("• " + inline(line[2:]), styles["bullet"]))
            i += 1
            continue

        # ── Italic/footer lines ───────────────────────────────────────────
        if line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            add(Paragraph(line[1:-1], styles["italic"]))
            i += 1
            continue

        # ── Blank lines ───────────────────────────────────────────────────
        if not line.strip():
            i += 1
            continue

        # ── Normal paragraph ──────────────────────────────────────────────
        add(body(line))
        i += 1

    return story


def build_pdf(md_path, pdf_path):
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2.0*cm,
        rightMargin=2.0*cm,
        topMargin=1.8*cm,
        bottomMargin=1.8*cm,
        title="Speed Safety Score — ADB Challenge 2026",
        author="hksamm / Pusan National University",
    )
    avail_w = W - 1.8*cm*2
    styles = make_styles()
    story = parse_md(md_path, styles, avail_w)
    doc.build(story)
    size = pdf_path.stat().st_size / 1024
    print(f"PDF saved → {pdf_path}  ({size:.0f} KB)")


if __name__ == "__main__":
    build_pdf(MD, PDF)
