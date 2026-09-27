"""Tool tests — research + image fallback chains (all hermetic)."""
import asyncio

from contentblitz import tools
from contentblitz.tools import (
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


def test_generate_image_offline_is_labelled_placeholder():
    result = asyncio.run(generate_image("a robot writing"))
    assert result["status"] == "placeholder"
    assert result["image_url"] is None
    assert "PLACEHOLDER" in result["note"]
    assert "not a real generated image" in result["note"]


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
    monkeypatch.setattr(tools, "_openai_image_sync", fake_gen)
    result = asyncio.run(generate_image("a robot"))
    assert result["status"] == "generated"
    assert result["model"] == "dall-e-2"
    assert attempts == ["dall-e-3", "dall-e-2"]


def test_generate_image_all_fail_is_placeholder(monkeypatch):
    import contentblitz.config as cfg

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_openai_image_sync", lambda p, m: None)
    result = asyncio.run(generate_image("a robot"))
    assert result["status"] == "placeholder"
    assert "PLACEHOLDER" in result["note"]


def test_check_image_support_no_key():
    support = check_image_support()
    assert support["key_present"] is False
    assert support["dall-e-3"] is False
