"""Export helpers — Markdown / HTML / PDF rendering of drafts.

Kept out of app.py so the logic is unit-testable without Streamlit.
"""
from __future__ import annotations

import html as html_lib
import re


def markdown_to_html(text: str) -> str:
    """Small safe Markdown subset -> HTML (headings, bold, links, lists)."""
    def inline(s: str) -> str:
        s = html_lib.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', s)
        return s

    out, in_list = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if stripped.startswith("### "):
                out.append(f"<h3>{inline(stripped[4:])}</h3>")
            elif stripped.startswith("## "):
                out.append(f"<h2>{inline(stripped[3:])}</h2>")
            elif stripped.startswith("# "):
                out.append(f"<h1>{inline(stripped[2:])}</h1>")
            elif stripped in ("---", "***"):
                out.append("<hr>")
            elif stripped:
                out.append(f"<p>{inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def export_html(drafts: dict) -> str:
    """Combine drafts into a standalone HTML document."""
    bodies = [f"# {kind.title()}\n\n{res['draft']}"
              for kind, res in drafts.items()
              if isinstance(res, dict) and res.get("draft")]
    combined = "\n\n---\n\n".join(bodies)
    body_html = markdown_to_html(combined)
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>ContentBlitz export</title></head><body>"
            f"{body_html}</body></html>")


def export_pdf(text: str) -> bytes:
    """Render plain-text export to PDF via fpdf2. Returns valid PDF bytes."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        stripped = line.strip()
        # NOTE: fpdf2 >= 2.8 leaves the cursor at the right margin after
        # multi_cell, so every call resets to the left margin explicitly.
        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 10, stripped[2:], new_x=XPos.LMARGIN,
                           new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 9, stripped[3:], new_x=XPos.LMARGIN,
                           new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=11)
        else:
            safe = stripped.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())
