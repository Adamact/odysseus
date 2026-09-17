from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preview_hides_import_action_and_restores_it_for_empty_editor():
    source = (ROOT / "static/js/document.js").read_text(encoding="utf-8")

    assert "const emptyImport = document.getElementById('doc-rich-empty-import');" in source
    assert "if (emptyImport) emptyImport.style.display = 'none';" in source
    assert "_syncRichEmptyImport(richEmailBody);" in source
