"""The `rag` UI toggle must be accepted.

do_ui_control advertises `rag` as a valid toggle in its own docstring and in
get_toggles ("Available toggles: web, bash, rag, ..."), and the frontend
fully wires it (chatStream.js maps rag -> rag-toggle / rag-indicator-btn).
But valid_toggles omitted "rag", so `toggle rag on` returned an "Unknown
toggle" error - the advertised capability was dead.
"""
import asyncio

from src.ai_interaction import do_ui_control
from routes import prefs_routes


def test_toggle_rag_on_is_accepted():
    r = asyncio.run(do_ui_control("toggle rag on"))
    assert r.get("ui_event") == "toggle"
    assert r.get("toggle_name") == "rag"
    assert r.get("state") is True
    assert "error" not in r


def test_toggle_rag_off_is_accepted():
    r = asyncio.run(do_ui_control("toggle rag off"))
    assert r.get("toggle_name") == "rag"
    assert r.get("state") is False
    assert "error" not in r


def test_unknown_toggle_still_rejected():
    r = asyncio.run(do_ui_control("toggle bogus on"))
    assert "error" in r


def test_existing_toggle_still_works():
    r = asyncio.run(do_ui_control("toggle web on"))
    assert r.get("toggle_name") == "web" and r.get("state") is True


def test_open_calendar_panel_is_accepted():
    r = asyncio.run(do_ui_control("open_panel calendar"))
    assert r.get("ui_event") == "open_panel"
    assert r.get("panel") == "calendar"
    assert "error" not in r


def test_open_calendar_panel_accepts_view_and_target_date():
    r = asyncio.run(do_ui_control("open_panel calendar month 2026-09"))
    assert r.get("ui_event") == "open_panel"
    assert r.get("panel") == "calendar"
    assert r.get("view") == "month"
    assert r.get("target_date") == "2026-09"
    assert "error" not in r


def test_models_panel_alias_opens_cookbook_models_view():
    r = asyncio.run(do_ui_control("open_panel models"))
    assert r.get("panel") == "cookbook"
    assert r.get("view") == "Search"
    assert r.get("view_label") == "models"
    assert "models view" in r.get("results", "")


def test_cookbook_panel_accepts_named_subview():
    r = asyncio.run(do_ui_control("open_panel cookbook serve"))
    assert r.get("panel") == "cookbook"
    assert r.get("view") == "Serve"
    assert r.get("view_label") == "launch"


def test_set_theme_persists_owner_scoped_name_for_later_verification(monkeypatch):
    stores = {"alice": {}}
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda owner: dict(stores.get(owner, {})))
    monkeypatch.setattr(prefs_routes, "_save_for_user", lambda owner, prefs: stores.__setitem__(owner, dict(prefs)))

    changed = asyncio.run(do_ui_control("set_theme dark", owner="alice"))
    current = asyncio.run(do_ui_control("get_theme", owner="alice"))

    assert changed.get("ui_event") == "set_theme"
    assert stores["alice"]["theme"] == {"name": "dark"}
    assert current == {
        "results": "Current theme: dark",
        "current_theme": "dark",
        "theme_known": True,
    }


def test_get_theme_does_not_invent_unsynchronized_client_state(monkeypatch):
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda owner: {})
    result = asyncio.run(do_ui_control("get_theme", owner="alice"))
    assert result["theme_known"] is False
    assert "not been synchronized" in result["results"].lower()
