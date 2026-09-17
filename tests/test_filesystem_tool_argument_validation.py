import pytest

from src.agent_tools.filesystem_tools import EditFileTool, ReadFileTool, WriteFileTool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "content", "error_fragment"),
    [
        (ReadFileTool(), '{"path":"broken"', "valid JSON"),
        (ReadFileTool(), "{}", "path required"),
        (WriteFileTool(), '{"path":"broken"', "valid JSON"),
        (WriteFileTool(), '{"path":"out.txt"}', "content required"),
        (
            EditFileTool(),
            '{"path":"a.txt","old_string":"x","new_string":"y","replace_all":"false"}',
            "must be a boolean",
        ),
    ],
)
async def test_filesystem_tools_reject_invalid_structured_arguments_before_disk_access(
    monkeypatch,
    tool,
    content,
    error_fragment,
):
    import src.tool_execution as tool_execution

    resolved = []

    def unexpected_resolve(path):
        resolved.append(path)
        raise AssertionError("invalid arguments reached path resolution")

    monkeypatch.setattr(tool_execution, "_resolve_tool_path", unexpected_resolve)

    result = await tool.execute(content, {})

    assert result["exit_code"] == 1
    assert error_fragment in result["error"]
    assert resolved == []


@pytest.mark.asyncio
async def test_write_file_preserves_existing_binary_artifact(tmp_path, monkeypatch):
    import src.tool_execution as tool_execution

    target = tmp_path / "output.pdf"
    original = b"%PDF-1.7\nvalid binary payload\x00\xff"
    target.write_bytes(original)
    monkeypatch.setattr(tool_execution, "_resolve_tool_path", lambda _path: str(target))

    result = await WriteFileTool().execute(
        '{"path":"output.pdf","content":"The PDF is already complete."}', {}
    )

    assert result["exit_code"] == 1
    assert result["binary_artifact_preserved"] is True
    assert "binary artifact path" in result["error"]
    assert target.read_bytes() == original


@pytest.mark.asyncio
async def test_write_file_rejects_new_binary_artifact_path(tmp_path, monkeypatch):
    import src.tool_execution as tool_execution

    target = tmp_path / "new.pdf"
    monkeypatch.setattr(tool_execution, "_resolve_tool_path", lambda _path: str(target))

    result = await WriteFileTool().execute(
        '{"path":"new.pdf","content":"not really a PDF"}', {}
    )

    assert result["exit_code"] == 1
    assert result["binary_artifact_preserved"] is False
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_file_still_rewrites_existing_text_file(tmp_path, monkeypatch):
    import src.tool_execution as tool_execution

    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(tool_execution, "_resolve_tool_path", lambda _path: str(target))

    result = await WriteFileTool().execute(
        '{"path":"notes.txt","content":"new"}', {}
    )

    assert result["exit_code"] == 0
    assert target.read_text(encoding="utf-8") == "new"
