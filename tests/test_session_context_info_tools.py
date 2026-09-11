from routes.session_routes import _context_info_tool_inventory


def test_context_info_tool_inventory_is_compact_backend_metadata():
    tools = _context_info_tool_inventory()
    names = {tool["name"] for tool in tools}

    assert "manage_skills" in names
    assert "ui_control" in names

    ui_control = next(tool for tool in tools if tool["name"] == "ui_control")
    assert ui_control["source"] == "backend"
    assert "description" in ui_control
    assert "schema" not in ui_control
    assert "parameters" not in ui_control


def test_context_info_tool_inventory_applies_limit():
    tools = _context_info_tool_inventory(limit=2)

    assert len(tools) == 2
