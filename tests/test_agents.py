"""Agent tests — all six specialists, hermetic offline mode.

The offline writers are deterministic templates; tests assert structure,
SEO artifacts, quality scores, citations, and honest labelling.
"""
import asyncio

from contentblitz import agents
from contentblitz.agents import (
    content_strategist,
    deep_research_agent,
    format_citations,
    image_agent,
    linkedin_writer,
    seo_blog_writer,
)
from contentblitz.memory import ConversationMemory

BRIEF = {"topic": "email marketing", "audience": "SaaS founders",
         "brand_voice": "professional"}


def test_deep_research_compiles_cited_pack():
    pack = asyncio.run(deep_research_agent(BRIEF))
    assert pack["topic"] == "email marketing"
    assert pack["provider"] == "mock"
    assert pack["citation_count"] >= 3
    assert pack["results"]
    assert "offline mode" in pack["summary"]


def test_deep_research_dedupes_urls(monkeypatch):
    import contentblitz.agents as agents_mod

    async def fake_research(query, max_results=None):
        return {"results": [
            {"title": "A", "url": "https://x.co/1", "snippet": "s",
             "source": "mock"},
            {"title": "B", "url": "https://x.co/1", "snippet": "dup",
             "source": "mock"},
        ], "provider": "mock", "query": query}

    monkeypatch.setattr(agents_mod, "web_research", fake_research)
    pack = asyncio.run(deep_research_agent(BRIEF))
    urls = [r["url"] for r in pack["results"]]
    assert len(urls) == len(set(urls))


def test_format_citations_labels_demo():
    research = {"results": [
        {"title": "T", "url": "https://x.co", "snippet": "s",
         "source": "mock"}]}
    assert "_(demo)_" in format_citations(research)


def test_seo_blog_writer_offline_structure():
    research = {"results": [
        {"title": "Benchmarks", "url": "https://x.co",
         "snippet": "open rates rose", "source": "mock"}],
        "summary": "summary"}
    out = asyncio.run(seo_blog_writer(BRIEF, research))
    assert out["kind"] == "blog"
    assert out["draft"].startswith("# ")
    assert "## " in out["draft"]
    assert "**Sources:**" in out["draft"]
    assert out["seo"]["keywords"]
    assert out["seo"]["meta_description"]
    assert out["seo"]["slug"] == "email-marketing"
    assert out["quality"]["score"] > 0
    assert out["offline"] is True


def test_seo_blog_writer_llm_path(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    async def fake_ask(system, user):
        return ("# Great Title\n\nIntro hook here.\n\n## Section A\n\n"
                "Body with a stat: 42% lift.\n\n## Section B\n\nMore body.\n\n"
                "## FAQ\n\n**Q?** A.\n\n## Conclusion\n\nDo it.")

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    out = asyncio.run(seo_blog_writer(BRIEF, {"results": [], "summary": ""}))
    assert out["draft"].startswith("# Great Title")
    assert out["offline"] is False


def test_linkedin_writer_offline():
    out = asyncio.run(linkedin_writer(BRIEF))
    assert out["kind"] == "linkedin"
    assert "?" in out["draft"]  # question CTA
    assert out["quality"]["word_count"] <= 400


def test_linkedin_writer_applies_feedback(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    seen = {}

    async def fake_ask(system, user):
        seen["user"] = user
        return "Short punchy post.\n\nAgree? #Growth"

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    asyncio.run(linkedin_writer(BRIEF, feedback="make it shorter"))
    assert "make it shorter" in seen["user"]


def test_image_agent_offline_placeholder_never_real(monkeypatch):
    import contentblitz.tools as tools_mod

    # Free provider down and no DALL-E key -> labelled placeholder.
    monkeypatch.setattr(tools_mod, "_pollinations_image_sync", lambda p: None)
    monkeypatch.setattr(tools_mod, "_sleep_between_attempts", lambda s: None)
    out = asyncio.run(image_agent(BRIEF))
    assert out["kind"] == "image"
    assert out["status"] == "placeholder"
    assert out["image_url"] is None
    assert "PLACEHOLDER" in out["note"]


def test_content_strategist_offline():
    out = asyncio.run(content_strategist(BRIEF))
    assert out["kind"] == "strategy"
    assert "Key messages" in out["draft"]
    assert "Repurposing" in out["draft"]
    assert out["quality"]["score"] > 0


def test_agents_registry_has_all_six():
    assert set(agents.AGENTS) == {
        "query_handler", "deep_research", "seo_blog",
        "linkedin", "image", "strategist",
    }


def test_agents_write_to_memory():
    mem = ConversationMemory()
    asyncio.run(seo_blog_writer(BRIEF, None, mem))
    assert mem.last_output("blog")
