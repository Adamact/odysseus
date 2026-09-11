"""Checklist follow-ups through the public tool and a disposable real database."""
import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import database
from src.tools.notes import do_manage_notes


@pytest.fixture
def checklist(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'checklists.db'}")
    database.Note.__table__.create(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    with factory() as db:
        db.add(database.Note(id="checklist-fixture", owner="sft_checklist", title="Fixture",
                             note_type="checklist", items=json.dumps([
                                 {"text": "tea", "done": False},
                                 {"text": "rice", "done": True},
                             ])))
        db.commit()
    yield "checklist-fixture"
    engine.dispose()


def call(note_id, action="view", owner="sft_checklist", **args):
    return asyncio.run(do_manage_notes(json.dumps({"action": action, "id": note_id, **args}), owner=owner))


def test_explicit_checked_state_is_idempotent(checklist):
    before = call(checklist)
    for _ in range(2):
        result = call(checklist, "toggle_item", index=1, done=True)
        assert result["exit_code"] == 0
        assert call(checklist) == before


@pytest.mark.parametrize("done", ["false", "true", 0, 1, None, [], {}])
def test_invalid_target_state_does_not_change_checklist(checklist, done):
    before = call(checklist)
    result = call(checklist, "toggle_item", index=1, done=done)
    assert result["exit_code"] == 1
    assert "boolean" in result["error"]
    assert call(checklist) == before


def test_explicit_uncheck_then_legacy_toggle(checklist):
    for _ in range(2):
        assert call(checklist, "toggle_item", index=1, done=False)["exit_code"] == 0
        assert "[ ] 1: rice" in call(checklist)["results"]
    assert call(checklist, "toggle_item", index=1)["exit_code"] == 0
    assert "[x] 1: rice" in call(checklist)["results"]
    assert call(checklist, "toggle_item", index=1)["exit_code"] == 0
    assert "[ ] 1: rice" in call(checklist)["results"]


@pytest.mark.parametrize("args", [{"index": -1}, {"index": 2}, {"owner": "other_owner", "index": 1}])
def test_invalid_index_or_owner_does_not_change_checklist(checklist, args):
    before = call(checklist)
    assert call(checklist, "toggle_item", done=False, **args)["exit_code"] == 1
    assert call(checklist) == before


def test_remove_item_never_silently_toggles_checked_state(checklist):
    before = call(checklist)
    result = call(checklist, "remove_item", index=1)
    assert result["exit_code"] == 1
    assert "update" in result["error"]
    assert call(checklist) == before
    # Follow the canonical replacement contract, then undo without losing state.
    assert call(checklist, "update", checklist_items=[{"text": "tea", "done": False}])["exit_code"] == 0
    assert "rice" not in call(checklist)["results"]
    assert call(checklist, "update", checklist_items=[
        {"text": "tea", "done": False}, {"text": "rice", "done": True},
    ])["exit_code"] == 0
    assert call(checklist) == before
