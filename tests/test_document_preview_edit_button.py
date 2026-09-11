"""Regression guards for the Markdown preview hover-to-edit control."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_JS = (ROOT / "static/js/document.js").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "static/style.css").read_text(encoding="utf-8")


def test_preview_installs_hover_edit_button():
    assert "function _installMarkdownPreviewEditButton(preview)" in DOC_JS
    assert "button.className = 'doc-preview-hover-edit';" in DOC_JS
    assert "_installMarkdownPreviewEditButton(preview);" in DOC_JS


def test_preview_edit_button_enters_write_mode_and_focuses_editor():
    assert "_setMarkdownPreviewActive(false, { remember: true });" in DOC_JS
    assert "document.getElementById('doc-editor-textarea')?.focus()" in DOC_JS


def test_preview_edit_button_is_hover_revealed_and_sticky():
    assert ".doc-md-preview:hover .doc-preview-hover-edit" in STYLE_CSS
    assert "position: sticky;" in STYLE_CSS
    assert "pointer-events: none;" in STYLE_CSS
