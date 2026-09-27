"""Query Handler tests — LLM router + deterministic offline fallback.

The offline fallback is fully deterministic (no key needed); the LLM
path is exercised with a stubbed _ask_llm and a fake key.
"""
import asyncio

from contentblitz.agents import route_query, route_query_sync
from contentblitz.memory import ConversationMemory


def test_sync_blog():
    d = route_query_sync("write a blog post about email marketing")
    assert d["intent"] == "blog"
    assert d["needs_research"] is True
    assert d["formats"] == ["blog"]


def test_sync_linkedin_no_research():
    d = route_query_sync("write a LinkedIn post about our launch")
    assert d["intent"] == "linkedin"
    assert d["needs_research"] is False


def test_sync_image():
    d = route_query_sync("generate an image for the launch visual")
    assert d["intent"] == "image"
    assert d["formats"] == ["image"]


def test_sync_research():
    d = route_query_sync("research content marketing trends and statistics")
    assert d["intent"] == "research"
    assert d["needs_research"] is True


def test_sync_strategy():
    d = route_query_sync("build a content strategy for Q4")
    assert d["intent"] == "strategy"
    assert d["formats"] == ["strategy"]


def test_sync_series():
    d = route_query_sync("plan a 5-part series for our launch campaign")
    assert d["intent"] == "series"
    assert d["needs_research"] is True


def test_sync_combo():
    d = route_query_sync("write a blog and a LinkedIn post about onboarding")
    assert d["intent"] == "combo"
    assert set(d["formats"]) == {"blog", "linkedin"}


def test_sync_research_plus_format_is_research_first_blog():
    d = route_query_sync("research the market and write a blog about pricing")
    assert d["intent"] == "blog"
    assert d["needs_research"] is True


def test_sync_chat():
    d = route_query_sync("hello, what can you do?")
    assert d["intent"] == "chat"
    assert d["needs_research"] is False


def test_sync_refine_with_draft():
    mem = ConversationMemory()
    mem.remember_output("blog", "some draft")
    d = route_query_sync("make it shorter and punchier", mem)
    assert d["intent"] == "refine"
    assert d["target_kind"] == "blog"


def test_sync_refine_without_draft_is_not_refine():
    mem = ConversationMemory()
    d = route_query_sync("make it shorter", mem)
    assert d["intent"] != "refine"  # no draft exists -> not a refinement


def test_async_uses_sync_offline():
    d = asyncio.run(route_query("write a blog post", ConversationMemory()))
    assert d["intent"] == "blog"


def test_async_llm_router_with_mocked_key(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    async def fake_ask(system, user):
        return ('{"intent": "linkedin", "needs_research": false, '
                '"formats": ["linkedin"], "confidence": 0.95, '
                '"reasoning": "test stub"}')

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    d = asyncio.run(route_query("write something social", ConversationMemory()))
    assert d["intent"] == "linkedin"
    assert d["confidence"] == 0.95


def test_async_llm_router_bad_json_falls_back(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    async def fake_ask(system, user):
        return "not json at all"

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    d = asyncio.run(route_query("write a blog post", ConversationMemory()))
    assert d["intent"] == "blog"  # deterministic fallback engaged
