from pathlib import Path


STYLE = Path(__file__).resolve().parents[1] / "static" / "style.css"


def test_mobile_sidebar_uses_the_sidebar_theme_surface():
    css = STYLE.read_text()
    mobile_drawer = css[css.index("/* Sidebar overlays chat on mobile */"):css.index("/* Backdrop behind sidebar */")]
    assert "background: var(--sidebar-bg, var(--panel)) !important;" in mobile_drawer
    assert "background: var(--panel) !important;" not in mobile_drawer
