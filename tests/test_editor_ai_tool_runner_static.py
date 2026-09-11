from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "static/js/editor/ai-tool-runner.js").read_text(encoding="utf-8")


def test_ai_runner_keeps_busy_state_until_result_image_is_decoded():
    assert "await new Promise((resolve, reject) =>" in RUNNER
    assert "img.onload = resolve;" in RUNNER
    assert "reject(new Error('Failed to decode result image'))" in RUNNER
    assert "layer.ctx.drawImage(img, 0, 0);" in RUNNER


def test_ai_runner_does_not_commit_a_result_from_a_closed_editor():
    assert "if (!state.editorOpen) return; // user closed mid-decode" in RUNNER
