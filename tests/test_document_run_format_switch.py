from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_JS = (ROOT / "static/js/document.js").read_text(encoding="utf-8")


def test_format_change_clears_stale_run_output() -> None:
    marker = "document.getElementById('doc-language-select').addEventListener('change', () => {"
    handler = DOCUMENT_JS.split(marker, 1)[1].split("    // Email send/draft buttons", 1)[0]

    assert "document.getElementById('doc-run-output')" in handler
    assert "staleRunOutput.style.display = 'none'" in handler
    assert "staleRunOutput.innerHTML = ''" in handler
