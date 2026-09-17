"""Entrypoint wiring: the processor contract is honoured before any LLM work."""
from __future__ import annotations

import pytest

from analyze import main as entry


@pytest.fixture(autouse=True)
def processor_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("LLM_GOVERNOR_FUNCTION_NAME", "governor-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:8765")
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)


def test_missing_sidecar_url_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_BASE_URL")
    assert entry.main() == 2
    assert "ANTHROPIC_BASE_URL" in capsys.readouterr().err


def test_missing_upstream_blackboard_exits_2(capsys):
    assert entry.main() == 2
    assert "Hypothesize did not produce" in capsys.readouterr().err


def test_analyze_model_pin_is_an_alias_of_the_synthesis_role():
    assert entry.ROLE_TABLE["synthesis"].env_aliases == ("VP_ANALYZE_MODEL",)
