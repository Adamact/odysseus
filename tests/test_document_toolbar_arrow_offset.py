from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_toolbar_arrows_have_real_flex_slots_outside_icon_scroller():
    script = (ROOT / "static/js/document.js").read_text()
    styles = (ROOT / "static/style.css").read_text()

    leading = script.index('class="md-toolbar-leading-controls"')
    left_arrow = script.index('id="md-scroll-left"')
    items = script.index('id="md-toolbar-items"')
    right_arrow = script.index('id="md-scroll-right"')
    assert leading < left_arrow < items < right_arrow
    assert "has-left-scroll-arrow" in script
    assert "has-right-scroll-arrow" in script
    assert ".md-toolbar-leading-controls" in styles
    assert "position: static" in styles
    assert "flex: 0 0 28px" in styles
