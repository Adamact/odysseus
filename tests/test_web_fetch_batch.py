import asyncio
import json

from src.agent_tools.web_tools import WebFetchTool
from src.search import content as content_mod
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block
from src.tool_execution import _active_workspace


def test_web_fetch_batch_is_concurrent_bounded_and_source_separated(monkeypatch):
    def fake_fetch(url, **_kwargs):
        return {
            "content": (f"evidence for {url} " * 1000),
            "title": url.rsplit("/", 1)[-1],
            "error": "",
        }

    monkeypatch.setattr(content_mod, "fetch_webpage_content", fake_fetch)
    result = asyncio.run(WebFetchTool().execute(json.dumps({
        "urls": [
            "https://example.com/a",
            "https://example.com/b",
        ],
    }), {}))

    assert result["exit_code"] == 0
    assert result["successful"] == 2
    assert result["requested"] == 2
    assert "## URL 1: https://example.com/a" in result["output"]
    assert "## URL 2: https://example.com/b" in result["output"]
    assert "[...batch item truncated]" in result["output"]


def test_web_fetch_batch_rejects_empty_or_oversized_lists():
    tool = WebFetchTool()
    empty = asyncio.run(tool.execute('{"urls": []}', {}))
    oversized = asyncio.run(tool.execute(json.dumps({
        "urls": [f"https://example.com/{index}" for index in range(13)],
    }), {}))
    assert empty["exit_code"] == 1
    assert "non-empty" in empty["error"]
    assert oversized["exit_code"] == 1
    assert "at most 12" in oversized["error"]


def test_web_fetch_batch_reads_local_workspace_text_files(tmp_path):
    (tmp_path / "one.json").write_text('{"one": 1}')
    (tmp_path / "two.json").write_text('{"two": 2}')
    token = _active_workspace.set(str(tmp_path))
    try:
        result = asyncio.run(WebFetchTool().execute(json.dumps({
            "urls": [
                "/workspace/one.json",
                "file:///workspace/two.json",
            ],
        }), {}))
    finally:
        _active_workspace.reset(token)

    assert result["exit_code"] == 0
    assert result["successful"] == 2
    assert '"one": 1' in result["output"]
    assert '"two": 2' in result["output"]
    assert "Read the local workspace path with read_file" in result["output"]


def test_web_fetch_does_not_decode_local_html_or_binary_media(tmp_path):
    token = _active_workspace.set(str(tmp_path))
    try:
        html = asyncio.run(WebFetchTool().execute("/workspace/page.html", {}))
        video = asyncio.run(WebFetchTool().execute("/workspace/video.mp4", {}))
    finally:
        _active_workspace.reset(token)

    assert html["exit_code"] == 1
    assert "private_browser" in html["error"]
    assert video["exit_code"] == 1
    assert "inspect_media" in video["error"]


def test_web_fetch_schema_and_native_parser_accept_urls_batch():
    schema = next(
        item["function"] for item in FUNCTION_TOOL_SCHEMAS
        if item["function"]["name"] == "web_fetch"
    )
    assert schema["parameters"]["properties"]["urls"]["maxItems"] == 12
    block = function_call_to_tool_block(
        "web_fetch",
        json.dumps({"urls": ["https://example.com/a", "https://example.com/b"]}),
    )
    assert block is not None
    assert block.tool_type == "web_fetch"


def test_web_fetch_parser_normalizes_empty_label_batch_pairs_only():
    block = function_call_to_tool_block(
        "web_fetch",
        json.dumps({"urls": [["https://example.com/a", ""]]}),
    )

    assert block is not None
    assert json.loads(block.content)["urls"] == ["https://example.com/a"]
