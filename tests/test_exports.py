"""Export tests — Markdown/HTML/PDF rendering of drafts."""
from contentblitz.exports import export_html, export_pdf, markdown_to_html


def test_markdown_to_html_renders_structure():
    md = ("# Title\n\n## Section\n\nSome **bold** text.\n\n"
          "- item one\n- item two\n\n[link](https://x.co)")
    html = markdown_to_html(md)
    assert "<h1>Title</h1>" in html
    assert "<h2>Section</h2>" in html
    assert "<strong>bold</strong>" in html
    assert "<ul>" in html and "<li>item one</li>" in html
    assert '<a href="https://x.co">link</a>' in html


def test_markdown_to_html_escapes_unsafe_input():
    html = markdown_to_html("<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_export_html_document():
    drafts = {"blog": {"draft": "# Hello\n\nBody text."},
              "empty": {"draft": ""}}
    doc = export_html(drafts)
    assert doc.startswith("<!DOCTYPE html>")
    assert "<h1>Blog</h1>" in doc  # kind title header
    assert "<h1>Hello</h1>" in doc  # draft content
    assert "Empty" not in doc  # drafts without text are skipped


def test_export_pdf_returns_valid_pdf():
    pdf = export_pdf("# Title\n\n## Section\n\nBody with unicode: caf\u00e9 \U0001f680")
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 500


def test_export_pdf_empty_input():
    pdf = export_pdf("")
    assert pdf[:5] == b"%PDF-"
