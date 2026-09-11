from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_image_settings_only_list_online_served_image_models():
    settings = (ROOT / "static/js/settings.js").read_text(encoding="utf-8")

    assert "fetch('/api/model-endpoints'" in settings
    assert "!endpoint.is_enabled || !endpoint.online" in settings
    assert "endpoint.model_type || '').toLowerCase() !== 'image'" in settings
    assert "stable-diffusion-3.5-medium', 'stable-diffusion-inpainting" not in settings
    assert "(not detected)" not in settings[settings.index("async function initImageSettings"):settings.index("function syncImgDisabled")]


def test_deep_research_fields_are_constrained_to_their_card():
    styles = (ROOT / "static/style.css").read_text(encoding="utf-8")

    rule = styles[styles.index("/* Deep Research uses long labels"):]
    assert ".admin-card:has(#set-researchSearch) .settings-row { min-width: 0; }" in rule
    assert "min-width: 0 !important;" in rule
    assert "flex: 1 1 0 !important;" in rule


def test_endpoint_probe_is_hidden_during_bulk_selection():
    admin = (ROOT / "static/js/admin.js").read_text(encoding="utf-8")

    update = admin[admin.index("function _updateEndpointBulkControls()"):admin.index("function _ensureEndpointBulkControls()")]
    assert "const probeBtn = el('adm-epProbeAllBtn');" in update
    assert "probeBtn.style.display = _endpointSelectMode ? 'none' : 'inline-flex'" in update
