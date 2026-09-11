import asyncio
import json
from types import SimpleNamespace

from PIL import Image

from src.tools.image import do_edit_image
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS


def test_edit_image_schema_forbids_dependency_install_fallbacks():
    schema = next(
        item["function"]
        for item in FUNCTION_TOOL_SCHEMAS
        if item.get("function", {}).get("name") == "edit_image"
    )

    description = schema["description"].lower()
    assert "missing optional dependency" in description
    assert "do not install packages" in description


class _Query:
    def __init__(self, source):
        self.source = source
        self.rejected = False

    def filter(self, *clauses):
        if False in clauses:
            self.rejected = True
        return self

    def first(self):
        return None if self.rejected else self.source


class _Db:
    def __init__(self, source):
        self.query_obj = _Query(source)
        self.added = []
        self.committed = False
        self.rolled_back = False

    def query(self, _model):
        return self.query_obj

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


def _source(filename):
    return SimpleNamespace(
        filename=filename,
        prompt="Source image",
        caption="Caption",
        quality="high",
        tags="reference",
        ai_tags="photo",
        session_id="session-1",
        album_id="album-1",
    )


def test_edit_image_upscale_creates_owned_copy(monkeypatch, tmp_path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (3, 2), "red").save(source_path)
    db = _Db(_source(source_path.name))

    monkeypatch.setattr("core.database.SessionLocal", lambda: db)
    monkeypatch.setattr("src.constants.GENERATED_IMAGES_DIR", str(tmp_path))

    result = asyncio.run(do_edit_image(json.dumps({
        "image_id": "source-id",
        "action": "upscale",
        "scale": 2,
    }), owner="alice"))

    assert result["exit_code"] == 0
    assert result["image_size"] == "6x4"
    assert source_path.exists()
    assert Image.open(source_path).size == (3, 2)
    assert len(db.added) == 1
    assert db.added[0].owner == "alice"
    assert db.added[0].filename != source_path.name
    assert (tmp_path / db.added[0].filename).exists()
    assert db.committed is True


def test_edit_image_without_owner_cannot_read_gallery(monkeypatch, tmp_path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (3, 2), "red").save(source_path)
    db = _Db(_source(source_path.name))

    monkeypatch.setattr("core.database.SessionLocal", lambda: db)
    monkeypatch.setattr("src.constants.GENERATED_IMAGES_DIR", str(tmp_path))

    result = asyncio.run(do_edit_image(
        '{"image_id":"source-id","action":"upscale","scale":2}',
        owner=None,
    ))

    assert result == {"error": "Image not found", "exit_code": 1}
    assert db.added == []


def test_edit_image_rejects_actions_without_complete_input_contract():
    result = asyncio.run(do_edit_image(
        '{"image_id":"source-id","action":"inpaint","prompt":"clouds"}',
        owner="alice",
    ))

    assert result["exit_code"] == 1
    assert "Use upscale or rembg" in result["error"]
