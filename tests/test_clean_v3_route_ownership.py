from routes.chat_routes import (
    _clean_v3_route_for_model,
    _native_runtime_requires_local_browser,
    _turn_contract_enabled,
)


def test_preheretic_model_owns_clean_route_independent_of_endpoint_alias():
    assert _clean_v3_route_for_model("odysseus-qwen3.5-tools-pre-heretic")


def test_other_models_keep_regular_harness():
    assert not _clean_v3_route_for_model("qwen35-9b-base")
    assert not _clean_v3_route_for_model("")


def test_clean_v3_keeps_contract_ownership_on_native_workspace():
    assert _turn_contract_enabled(
        exact_tool_approval=None,
        runtime_surface="odysseus-native",
        native_workspace_contract=True,
        clean_v3_route=True,
    )


def test_other_models_keep_separate_native_workspace_contract():
    assert not _turn_contract_enabled(
        exact_tool_approval=None,
        runtime_surface="odysseus-native",
        native_workspace_contract=True,
        clean_v3_route=False,
    )


def test_tui_and_exact_approval_still_bypass_routed_contract():
    assert not _turn_contract_enabled(
        exact_tool_approval=None,
        runtime_surface="odysseus-tui",
        native_workspace_contract=False,
        clean_v3_route=True,
    )


def test_native_html_artifact_requires_local_browser_without_public_web():
    context = {
        "surface": "odysseus-native",
        "terminal_agent": True,
        "unattended_mode": True,
        "completion_requirements": {
            "required_artifacts": ["/workspace/output.html"],
        },
    }
    assert _native_runtime_requires_local_browser(context)
    context["completion_requirements"]["required_artifacts"] = ["/workspace/output.txt"]
    assert not _native_runtime_requires_local_browser(context)
    assert not _turn_contract_enabled(
        exact_tool_approval=object(),
        runtime_surface="",
        native_workspace_contract=False,
        clean_v3_route=True,
    )
