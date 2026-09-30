"""Tool tests — research + image fallback chains (all hermetic)."""
import asyncio

from contentblitz import tools
from contentblitz.tools import (
    _pollinations_image_sync,
    check_image_support,
    generate_image,
    optimize_image_prompt,
    web_research,
)


def test_web_research_offline_returns_labelled_mock():
    pack = asyncio.run(web_research("email marketing"))
    assert pack["provider"] == "mock"
    assert len(pack["results"]) >= 3
    assert all(r["source"] == "mock" for r in pack["results"])
    assert all(r["url"] for r in pack["results"])


def test_web_research_tavily_preferred(monkeypatch):
    import contentblitz.config as cfg

    calls = []

    def fake_tavily(query, n):
        calls.append("tavily")
        return [{"title": "T", "url": "https://t.co", "snippet": "s",
                 "source": "tavily"}]

    def fake_serpapi(query, n):
        calls.append("serpapi")
        return [{"title": "S", "url": "https://s.co", "snippet": "s",
                 "source": "serpapi"}]

    monkeypatch.setattr(cfg.settings, "TAVILY_API_KEY", "tv-key")
    monkeypatch.setattr(cfg.settings, "SERPAPI_API_KEY", "sp-key")
    monkeypatch.setattr(tools, "_tavily_search_sync", fake_tavily)
    monkeypatch.setattr(tools, "_serpapi_search_sync", fake_serpapi)
    pack = asyncio.run(web_research("pricing"))
    assert pack["provider"] == "tavily"
    assert calls == ["tavily"]  # fallback never engaged


def test_web_research_serpapi_fallback(monkeypatch):
    import contentblitz.config as cfg

    def fake_tavily(query, n):
        return None  # Tavily failed

    def fake_serpapi(query, n):
        return [{"title": "S", "url": "https://s.co", "snippet": "s",
                 "source": "serpapi"}]

    monkeypatch.setattr(cfg.settings, "TAVILY_API_KEY", "tv-key")
    monkeypatch.setattr(cfg.settings, "SERPAPI_API_KEY", "sp-key")
    monkeypatch.setattr(tools, "_tavily_search_sync", fake_tavily)
    monkeypatch.setattr(tools, "_serpapi_search_sync", fake_serpapi)
    pack = asyncio.run(web_research("pricing"))
    assert pack["provider"] == "serpapi"


def test_web_research_both_fail_gives_mock(monkeypatch):
    import contentblitz.config as cfg

    monkeypatch.setattr(cfg.settings, "TAVILY_API_KEY", "tv-key")
    monkeypatch.setattr(cfg.settings, "SERPAPI_API_KEY", "sp-key")
    monkeypatch.setattr(tools, "_tavily_search_sync", lambda q, n: None)
    monkeypatch.setattr(tools, "_serpapi_search_sync", lambda q, n: None)
    pack = asyncio.run(web_research("pricing"))
    assert pack["provider"] == "mock"
    assert all(r["source"] == "mock" for r in pack["results"])


def test_optimize_image_prompt_includes_brief_fields():
    brief = {"topic": "AI analytics", "audience": "CTOs",
             "image_style": "isometric 3D", "mood": "futuristic",
             "brand": "Acme"}
    prompt = optimize_image_prompt(brief)
    assert "AI analytics" in prompt
    assert "CTOs" in prompt
    assert "isometric 3D" in prompt
    assert "no readable text" in prompt


def test_generate_image_offline_is_labelled_placeholder(monkeypatch):
    # Free provider down and no DALL-E key -> labelled placeholder.
    monkeypatch.setattr(tools, "_pollinations_image_sync", lambda p: None)
    monkeypatch.setattr(tools, "_sleep_between_attempts", lambda s: None)
    result = asyncio.run(generate_image("a robot writing"))
    assert result["status"] == "placeholder"
    assert result["image_url"] is None
    assert "PLACEHOLDER" in result["note"]
    assert "not a real generated image" in result["note"]


def test_generate_image_pollinations_first_success(monkeypatch):
    import contentblitz.config as cfg

    def fake_get(url, timeout=None):
        class _Resp:
            status_code = 200
            headers = {"content-type": "image/jpeg"}

        assert url.startswith("https://image.pollinations.ai/prompt/")
        assert "model=flux" in url
        return _Resp()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    result = asyncio.run(generate_image("a robot"))
    assert result["status"] == "generated"
    assert result["model"] == "pollinations/flux"
    assert result["image_url"].startswith("https://image.pollinations.ai/")


def test_pollinations_image_sync_success(monkeypatch):
    def fake_get(url, timeout=None):
        class _Resp:
            status_code = 200
            headers = {"content-type": "image/png"}

        return _Resp()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    out = _pollinations_image_sync("a robot painting")
    assert out["status"] == "generated"
    assert out["model"] == "pollinations/flux"
    assert "a%20robot%20painting" in out["image_url"]


def test_pollinations_image_sync_rejects_non_image(monkeypatch):
    def fake_get(url, timeout=None):
        class _Resp:
            status_code = 200
            headers = {"content-type": "text/html"}

        return _Resp()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    assert _pollinations_image_sync("a robot") is None


def test_pollinations_image_sync_request_failure(monkeypatch):
    def boom(url, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(tools.requests, "get", boom)
    assert _pollinations_image_sync("a robot") is None


def test_generate_image_dalle3_then_dalle2(monkeypatch):
    import contentblitz.config as cfg

    attempts = []

    def fake_gen(prompt, model):
        attempts.append(model)
        if model == "dall-e-3":
            return None  # primary fails
        return {"status": "generated", "image_url": "https://img.co/1",
                "model": model, "prompt": prompt}

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_pollinations_image_sync", lambda p: None)
    monkeypatch.setattr(tools, "_sleep_between_attempts", lambda s: None)
    monkeypatch.setattr(tools, "_openai_image_sync", fake_gen)
    result = asyncio.run(generate_image("a robot"))
    assert result["status"] == "generated"
    assert result["model"] == "dall-e-2"
    assert attempts == ["dall-e-3", "dall-e-2"]


def test_generate_image_all_fail_is_placeholder(monkeypatch):
    import contentblitz.config as cfg

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_pollinations_image_sync", lambda p: None)
    monkeypatch.setattr(tools, "_sleep_between_attempts", lambda s: None)
    monkeypatch.setattr(tools, "_openai_image_sync", lambda p, m: None)
    result = asyncio.run(generate_image("a robot"))
    assert result["status"] == "placeholder"
    assert "PLACEHOLDER" in result["note"]


def test_generate_image_pollinations_retries_then_succeeds(monkeypatch):
    """HTTP 500 twice then 200: retries happen and success is returned.

    Regression test for the real end-user failure where the first two
    Pollinations.ai attempts returned HTTP 500 and the app fell straight
    through to the labelled placeholder without retrying.
    """
    import contentblitz.config as cfg

    calls = []
    sleeps = []

    def fake_get(url, timeout=None):
        calls.append(url)

        class _Resp:
            status_code = 500 if len(calls) < 3 else 200
            headers = {"content-type": "image/jpeg"}

        return _Resp()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    monkeypatch.setattr(tools, "_sleep_between_attempts", sleeps.append)
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    result = asyncio.run(generate_image("a robot"))
    assert len(calls) == 3  # retried until it succeeded
    assert sleeps == [1.0, 2.0]  # exponential backoff: 1s, then 2s
    assert result["status"] == "generated"
    assert result["model"] == "pollinations/flux"
    assert result["image_url"].startswith("https://image.pollinations.ai/")


def test_generate_image_pollinations_three_failures_falls_back(monkeypatch):
    """Three failed Pollinations attempts -> stop retrying, fall through.

    With no OpenAI key present the chain ends at the labelled placeholder,
    which must stay honestly labelled (never presented as a real image).
    """
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)

        class _Resp:
            status_code = 500
            headers = {"content-type": "image/jpeg"}

        return _Resp()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    monkeypatch.setattr(tools, "_sleep_between_attempts", lambda s: None)
    result = asyncio.run(generate_image("a robot"))
    assert len(calls) == 3  # exactly 3 attempts, then give up on Pollinations
    assert result["status"] == "placeholder"
    assert result["image_url"] is None
    assert result["model"] is None
    assert "PLACEHOLDER" in result["note"]
    assert "not a real generated image" in result["note"]


def test_check_image_support_no_key():
    support = check_image_support()
    assert support["key_present"] is False
    assert support["dall-e-3"] is False
    # Pollinations is free and keyless, so it is always available.
    assert support["pollinations"] is True
