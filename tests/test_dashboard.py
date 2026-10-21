from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).parents[1] / "dashboard"
PAGES = ["executive", "clients", "clienteling", "planning", "tier_dial", "health", "method"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_from_the_public_bundle(page: str) -> None:
    script = f"import sys\nsys.path.insert(0, {str(DASHBOARD)!r})\nimport app\napp.style()\napp.{page}()\n"
    at = AppTest.from_string(script, default_timeout=60).run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.title, f"{page} has no title"


def test_tier_dial_controls_change_the_view() -> None:
    script = f"import sys\nsys.path.insert(0, {str(DASHBOARD)!r})\nimport app\napp.tier_dial()\n"
    at = AppTest.from_string(script, default_timeout=60).run()
    at.session_state["brand_cost"] = "20%"
    at.session_state["lift_scenario"] = "conservative"
    at.run()

    assert not at.exception
