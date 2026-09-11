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
