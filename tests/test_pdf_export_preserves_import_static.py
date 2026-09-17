from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pdf_backed_documents_do_not_offer_destructive_html_pdf_export():
    source = (ROOT / "static/js/document.js").read_text(encoding="utf-8")

    assert "if (!isForm) {" in source
    assert "label: _isDocxLang(lang) ? 'Convert to PDF' : 'Print as PDF'" in source
    assert "destroy the original page layout, images, and form structure" in source
