"""Smoke tests for the Streamlit app.

These run the real script through Streamlit's AppTest harness, so they catch
the class of breakage that only appears when the page actually renders — a
NameError in a rarely-taken branch, a missing import inside a method, a
malformed dataframe — which neither a syntax gate nor an import check will
find.

Deliberately light on assertions about wording: these check that the page
renders without raising and that the key surfaces are present.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("rdkit", reason="rdkit is required")
pytest.importorskip("streamlit", reason="streamlit is required")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "integrated_multiagent_platform.py")


def all_text(at) -> str:
    """Every rendered text surface, since Streamlit splits them by element type."""
    parts = []
    for attr in ("markdown", "caption", "subheader", "header", "title",
                 "success", "warning", "info", "error"):
        for el in getattr(at, attr, []):
            value = getattr(el, "value", None)
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts)

# The app loads five Random Forest models plus their fingerprint caches on
# import, which takes a few seconds on a cold process.
TIMEOUT = 180


@pytest.fixture(scope="module")
def landing():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    return at


def test_app_renders_without_exception(landing):
    assert not landing.exception, (
        f"app raised on load: {[str(e) for e in landing.exception]}"
    )


def test_landing_page_shows_capabilities(landing):
    assert "Platform Capabilities" in all_text(landing)


def test_landing_page_advertises_the_new_features(landing):
    """The new work has to be discoverable, not just present."""
    markdown = all_text(landing)
    for feature in ("Novelty", "Selectivity", "Batch Screening"):
        assert feature in markdown, f"{feature} is not surfaced on the landing page"


def test_agent_card_style_is_defined(landing):
    """`agent-card` was used on the landing page but never defined in the CSS,
    so those cards rendered unstyled."""
    markdown = all_text(landing)
    if "agent-card" not in markdown:
        pytest.skip("agent cards not rendered")
    assert ".agent-card" in markdown, (
        "the agent-card class is used but has no CSS rule"
    )


def test_analysis_modes_include_batch_screening(landing):
    assert landing.sidebar.radio, "no analysis-mode selector rendered"
    options = landing.sidebar.radio[0].options
    assert "Batch Screening" in options


def test_every_analysis_mode_renders(landing):
    """Switch through each standalone mode and confirm none raises."""
    modes = landing.sidebar.radio[0].options

    for mode in modes:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        at.sidebar.radio[0].set_value(mode).run()
        assert not at.exception, (
            f"mode {mode!r} raised: {[str(e) for e in at.exception]}"
        )


def test_model_transparency_reports_scaffold_numbers(landing):
    """The honest generalization figure must be on the page, not just the
    optimistic random-split one."""
    combined = all_text(landing)

    if "Model Performance" not in combined:
        pytest.skip("QSAR models unavailable, transparency panel not shown")

    assert "scaffold" in combined.lower(), (
        "scaffold-split performance is not reported on the landing page"
    )


def test_system_status_toggle_renders(landing):
    """Toggling the sidebar status must not raise."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    boxes = [c for c in at.sidebar.checkbox if "status" in c.label.lower()]
    if not boxes:
        pytest.skip("status toggle not present")
    boxes[0].set_value(True).run()
    assert not at.exception
