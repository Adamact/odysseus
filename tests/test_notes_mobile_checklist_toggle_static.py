from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent


def test_mobile_notes_checklist_rows_remain_tappable():
    css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    notes = (ROOT / "static" / "js" / "notes.js").read_text(encoding="utf-8")

    assert "body.notes-mobile-mode .note-card .note-checkbox {" in css
    assert "pointer-events: auto;" in css
    assert "touch-action: manipulation;" in css
    assert "body.notes-mobile-mode .note-card .note-checkbox,\nbody.notes-mobile-mode .note-card .note-checkbox-rm" not in css
    assert "if (_selectMode) return; // let card-level handler take over\n      e.preventDefault();" in notes


def test_mobile_edit_checklist_toggle_marks_form_dirty():
    notes = (ROOT / "static" / "js" / "notes.js").read_text(encoding="utf-8")

    assert "container.dispatchEvent(new Event('input', { bubbles: true }));" in notes


def test_completed_mobile_todo_uses_finish_action():
    notes = (ROOT / "static" / "js" / "notes.js").read_text(encoding="utf-8")

    assert "const _isFinishedTodo = () =>" in notes
    assert "const label = _isFinishedTodo() ? 'Finish' : 'Archive';" in notes
    assert "if (_isFinishedTodo()) _enterArchive();" in notes


def test_mobile_long_press_enters_select_mode_for_that_note():
    notes = (ROOT / "static" / "js" / "notes.js").read_text(encoding="utf-8")

    assert "function _enterSelectMode(initialId = null)" in notes
    assert "_suppressNextNoteTapId = card.dataset.noteId;" in notes
    assert "_enterSelectMode(card.dataset.noteId);" in notes


def test_mobile_select_mode_has_subtle_jiggle_feedback():
    css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    assert "@keyframes notes-select-jiggle" in css
    assert "body.notes-mobile-mode .note-card-selectmode" in css
    assert "prefers-reduced-motion: reduce" in css


def test_notes_mobile_checklist_asset_versions_are_bumped():
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert re.search(r"notes\.js\?v=[A-Za-z0-9_-]+", app)
    app_versions = re.findall(r"/static/app\.js\?v=([A-Za-z0-9_-]+)", html)
    style_version = re.search(r"/static/style\.css\?v=([A-Za-z0-9_-]+)", html)
    assert app_versions and len(set(app_versions)) == 1
    assert style_version and style_version.group(1) == app_versions[0]


def test_drawing_edits_mark_notes_dirty_and_keep_one_gallery_image():
    notes = (ROOT / "static" / "js" / "notes.js").read_text(encoding="utf-8")
    upload = (ROOT / "routes" / "upload_routes.py").read_text(encoding="utf-8")
    note_routes = (ROOT / "routes" / "note" / "note_routes.py").read_text(encoding="utf-8")

    assert "const _markCanvasDirty = () => container.dispatchEvent(new Event('input', { bubbles: true }));" in notes
    assert "_uploadCanvasAsPng(canvas, note?.gallery_id || null)" in notes
    assert "fd.append('gallery_id', galleryId)" in notes
    assert "if gallery_id:" in upload
    assert "note.gallery_id = body.gallery_id" in note_routes


def test_calendar_email_attachments_have_a_calendar_import_action():
    email = (ROOT / "static" / "js" / "emailLibrary.js").read_text(encoding="utf-8")
    calendar = (ROOT / "static" / "js" / "calendar.js").read_text(encoding="utf-8")

    assert "email-attachment-calendar-open" in email
    assert "const _CALENDAR_RE = /\\.(calendar|ics|ical)$/i" in email
    assert "fetch(`${API_BASE}/api/calendar/import`" in email
    assert 'accept=".calendar,.ics,.ical"' in calendar


def test_mobile_bulk_select_long_press_is_shared_across_card_types():
    helper = (ROOT / "static" / "js" / "mobileBulkSelect.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    assert ".memory-item[data-memory-id]" in helper
    assert ".skill-card[data-skill-name]" in helper
    assert ".task-card[data-id]" in helper
    assert "const HOLD_MS = 450" in helper
    assert "body:has(#memory-select-btn.active)" in css
    assert "body:has(#skills-select-btn.active)" in css
    assert "body:has(#tasks-select-btn.active)" in css
