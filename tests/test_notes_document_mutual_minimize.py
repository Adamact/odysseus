from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_opening_notes_minimizes_open_document():
    script = (ROOT / "static/js/notes.js").read_text()

    assert "window.documentModule?.isPanelOpen?.()" in script
    assert "window.documentModule.closePanel('down')" in script


def test_opening_document_minimizes_notes():
    script = (ROOT / "static/js/document.js").read_text()

    assert "function _minimizeNotesForDocumentOpen()" in script
    assert "window.notesModule.closePanel('down')" in script
    assert "close('down')" in script
    assert "_minimizeNotesForDocumentOpen();" in script
