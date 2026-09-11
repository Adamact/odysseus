from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent


def test_email_folder_and_filter_pickers_are_exclusive_and_escape_safe():
    source = (ROOT / "static/js/emailLibrary.js").read_text(encoding="utf-8")

    assert source.count("const filterMenu = document.getElementById('email-filter-menu');") == 1
    assert source.count("const folderMenu = document.getElementById('email-folder-menu');") == 1
    assert source.count("e.stopImmediatePropagation();") >= 2
    assert "_wireEmailPickerEscapeGuard();" in source
    assert "window.addEventListener('keydown'" in source
    inbox = (ROOT / "static/js/emailInbox.js").read_text(encoding="utf-8")
    assert re.search(r"from './emailLibrary\.js\?v=[A-Za-z0-9_-]+'", inbox)
