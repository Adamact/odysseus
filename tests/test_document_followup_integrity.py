"""Document tool follow-ups against disposable real SQLite storage."""
import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import database
from src import database as compatibility_database
from src.agent_tools import TOOL_HANDLERS
from src.agent_tools.document_tools import get_active_document, set_active_document


@pytest.fixture
def documents(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'documents.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.setattr(compatibility_database, "SessionLocal", factory)
    saved_active = get_active_document()
    set_active_document(None)
    with factory() as db:
        db.add(database.Document(id="owned-document", owner="fixture-owner", title="Owned",
            language="text", current_content="First: alpha\nSecond: alpha\nKeep: violet-72"))
        db.add(database.Document(id="foreign-document", owner="other-owner", title="Foreign",
            language="text", current_content="Foreign alpha"))
        db.commit()
    yield
    set_active_document(saved_active)
    engine.dispose()


def read(document_id="owned-document", owner="fixture-owner"):
    return asyncio.run(TOOL_HANDLERS["manage_documents"](
        json.dumps({"action": "read", "document_id": document_id}), {"owner": owner}))


def edit(find, replacement, **ctx):
    return asyncio.run(TOOL_HANDLERS["edit_document"](
        f"<<<FIND>>>\n{find}\n<<<REPLACE>>>\n{replacement}\n<<<END>>>",
        {"owner": "fixture-owner", **ctx}))


def test_missing_explicit_target_does_not_edit_another_document(documents):
    before = read()
    result = edit("alpha", "WRONG", doc_id="deleted-document")
    assert "error" in result
    assert read() == before


def test_missing_explicit_target_does_not_replace_another_document(documents):
    before = read()
    result = asyncio.run(TOOL_HANDLERS["update_document"]("WRONG replacement", {
        "owner": "fixture-owner", "doc_id": "deleted-document"}))
    assert "error" in result
    assert read() == before


def test_missing_explicit_target_does_not_delete_another_document(documents):
    before = read()
    result = asyncio.run(TOOL_HANDLERS["manage_documents"](
        json.dumps({"action": "delete", "document_id": "deleted-document"}), {"owner": "fixture-owner"}))
    assert result.get("exit_code") == 1
    assert read() == before


@pytest.mark.parametrize("tool", ["edit_document", "update_document", "manage_documents"])
@pytest.mark.parametrize("target", ["deleted-document", "foreign-document"])
def test_unavailable_active_target_never_falls_back_to_other_document(documents, tool, target):
    before = read()
    foreign_before = read("foreign-document", "other-owner")
    set_active_document(target)
    content = {"edit_document": "<<<FIND>>>\nalpha\n<<<REPLACE>>>\nWRONG\n<<<END>>>",
               "update_document": "WRONG", "manage_documents": '{"action":"delete"}'}[tool]
    result = asyncio.run(TOOL_HANDLERS[tool](content, {"owner": "fixture-owner"}))
    assert result.get("exit_code") == 1
    assert read() == before
    assert read("foreign-document", "other-owner") == foreign_before


def test_targeted_edit_and_undo_preserve_other_occurrences(documents):
    before = read()
    result = edit("Second: alpha", "Second: beta", doc_id="owned-document")
    assert result["applied"] == 1 and result["skipped"] == 0
    assert read()["document"]["content"] == "First: alpha\nSecond: beta\nKeep: violet-72"
    assert "error" not in edit("Second: beta", "Second: alpha", doc_id="owned-document")
    assert read() == before


def test_no_target_legacy_fallback_still_scopes_to_owner(documents):
    set_active_document(None)
    result = edit("Second: alpha", "Second: beta")
    assert result["doc_id"] == "owned-document"
    assert "Second: beta" in read()["document"]["content"]


def test_sealed_missing_target_still_reports_version_guard(documents):
    before = read()
    result = edit("alpha", "WRONG", doc_id="deleted-document", expected_document_version=1)
    assert "error" in result
    assert read() == before


def test_delete_respects_dispatch_target_instead_of_global_active_document(documents):
    before = read()
    result = asyncio.run(TOOL_HANDLERS["manage_documents"]('{"action":"delete"}', {
        "owner": "fixture-owner", "doc_id": "deleted-document"}))
    assert result.get("exit_code") == 1
    assert read() == before


def test_delete_rejects_a_changed_sealed_document_version(documents):
    before = read()
    result = asyncio.run(TOOL_HANDLERS["manage_documents"](
        '{"action":"delete","document_id":"owned-document"}', {
            "owner": "fixture-owner", "doc_id": "owned-document", "expected_document_version": 99}))
    assert "error" in result
    assert read() == before


def test_valid_dispatch_delete_uses_its_target_and_matching_version(documents):
    foreign_before = read("foreign-document", "other-owner")
    result = asyncio.run(TOOL_HANDLERS["manage_documents"]('{"action":"delete"}', {
        "owner": "fixture-owner", "doc_id": "owned-document", "expected_document_version": 1}))
    assert result["exit_code"] == 0
    assert read()["exit_code"] == 1
    assert read("foreign-document", "other-owner") == foreign_before


def test_partial_multi_edit_reports_skipped_changes_without_claiming_all_applied(documents):
    from src.tool_execution import format_tool_result
    result = asyncio.run(TOOL_HANDLERS["edit_document"](
        '<<<FIND>>>\nSecond: alpha\n<<<REPLACE>>>\nSecond: beta\n<<<END>>>\n'
        '<<<FIND>>>\nAbsent text\n<<<REPLACE>>>\nWrong\n<<<END>>>',
        {"owner": "fixture-owner", "doc_id": "owned-document"}))
    assert result["applied"] == 1 and result["skipped"] == 1
    assert '"skipped": 1' in format_tool_result('edit_document', result)
    assert read()["document"]["content"] == "First: alpha\nSecond: beta\nKeep: violet-72"
