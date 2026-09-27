"""Hermetic fixtures: force offline mode unless a test opts into live keys."""
import pytest

import contentblitz.config as cfg


@pytest.fixture(autouse=True)
def offline_mode(monkeypatch, request):
    """Blank all API keys so hermetic tests run deterministically offline.

    Live integration tests (test_live_integrations.py) are exempt — they
    skip themselves when keys are absent.
    """
    if "live_integrations" in str(request.fspath):
        yield
        return
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(cfg.settings, "TAVILY_API_KEY", "")
    monkeypatch.setattr(cfg.settings, "SERPAPI_API_KEY", "")
    yield
