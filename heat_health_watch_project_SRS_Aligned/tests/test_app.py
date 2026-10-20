import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import config as C  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(C.PROJECT_ROOT / "app.py"), default_timeout=180).run()


def _click(at, key):
    [b for b in at.button if b.key == key][0].click()
    return at.run()


def test_app_starts_without_exceptions(app):
    assert not app.exception and not app.error
    assert len(app.tabs) == 8


def test_endpoint_banner_shows_last_date(app):
    assert any("30 November 2024" in i.value for i in app.info)


def test_calculator_button_returns_result(app):
    at = _click(app, "calc_btn")
    assert not at.exception and at.session_state["calc"]["score"] > 0


def test_rf_prediction_button(app):
    at = _click(app, "rf_btn")
    assert not at.exception and at.session_state["rf_pred"][0]["predicted_label"] in {"Moderate", "High"}


def test_forecast_button_returns_rows(app):
    at = _click(app, "f_btn")
    assert not at.exception and len(at.session_state["forecast"]) == 14


def test_forecast_invalid_range_shows_error():
    at = AppTest.from_file(str(C.PROJECT_ROOT / "app.py"), default_timeout=180).run()
    start_widget = [d for d in at.date_input if str(d.key).startswith("f_start_")][0]
    end_widget = [d for d in at.date_input if str(d.key).startswith("f_end_")][0]
    # Both dates remain after the synthetic extension, but the end precedes the start.
    start_widget.set_value(end_widget.value + __import__("datetime").timedelta(days=10))
    end_widget.set_value(end_widget.value + __import__("datetime").timedelta(days=5))
    at = at.run()
    at = _click(at, "f_btn")
    assert any("not calculated" in e.value for e in at.error)
    assert "forecast" not in at.session_state


def test_hourly_button(app):
    at = _click(app, "h_btn")
    assert not at.exception and len(at.session_state["hourly"]) == 24


def test_synthetic_button(app):
    at = _click(app, "syn_btn")
    assert not at.exception and len(at.session_state["syn"]) == 50
