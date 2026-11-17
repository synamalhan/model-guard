from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_dashboard_data import db_file  # noqa: F401  (module-scoped fixture)

APP = str(Path(__file__).parent.parent / "src/modelguard/dashboard.py")


def test_app_renders_alert_feed(db_file, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MODELGUARD_DB", str(db_file))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("alert(s) raised" in w.value for w in at.warning)
    assert len(at.dataframe) >= 1


def test_app_handles_empty_database(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELGUARD_DB", str(tmp_path / "empty.db"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("No runs" in e.value for e in at.error)
