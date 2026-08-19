from __future__ import annotations

import pytest

from analyze import models

ALLOW_LIST = [
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]


class Denied(Exception):
    def __init__(self, body):
        self.body = body
        super().__init__(repr(body))


def test_reported_allow_list_routes_analyze_roles():
    resolver = models.ModelResolver(ALLOW_LIST)
    assert resolver.resolve("opus") == "us.anthropic.claude-sonnet-4-6"
    assert resolver.resolve("sonnet") == "us.anthropic.claude-sonnet-4-6"


def test_allow_list_is_parsed_from_governor_error():
    error = Denied({"error": "model_not_allowed", "allowedModels": ALLOW_LIST})
    assert models.is_model_not_allowed(error)
    assert models.parse_allowed_models(error) == ALLOW_LIST


def test_legacy_analyze_model_override_is_a_validated_pin(monkeypatch):
    monkeypatch.setenv(
        "VP_ANALYZE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    resolver = models.ModelResolver(ALLOW_LIST)
    assert resolver.resolve("sonnet") == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert resolver.assignments()["sonnet"]["source"] == "pinned"


def test_conflicting_analyze_model_overrides_fail_clearly(monkeypatch):
    monkeypatch.setenv("VP_MODEL_SONNET", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv(
        "VP_ANALYZE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    with pytest.raises(models.ModelNotAvailable, match="Conflicting model pins"):
        models.ModelResolver(ALLOW_LIST).resolve("sonnet")
