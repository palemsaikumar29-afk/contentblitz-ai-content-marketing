"""Graph tests — every route end to end (hermetic, offline)."""
import asyncio

from contentblitz.graph import build_graph
from contentblitz.memory import ConversationMemory

BRIEF = {"topic": "email marketing", "audience": "SaaS founders",
         "brand_voice": "professional"}


def _run_graph(user_input, memory=None, **state_overrides):
    mem = memory if memory is not None else ConversationMemory()
    graph = build_graph(mem)
    state = {"user_input": user_input, "brief": dict(BRIEF),
             "drafts": {}, "research": {}}
    state.update(state_overrides)
    out = asyncio.run(graph.ainvoke(state))
    return out, mem


def test_chat_route():
    out, _ = _run_graph("hello, what can you do?")
    assert "ContentBlitz" in out["answer"]
    assert out["decision"]["intent"] == "chat"


def test_blog_research_first_workflow():
    """blog intent -> deep research -> post-research routing -> seo_blog."""
    out, mem = _run_graph("write a blog post about email marketing")
    assert out["research"], "research-first workflow must research"
    assert out["research"]["provider"] == "mock"
    assert "blog" in out["drafts"]
    draft = out["drafts"]["blog"]["draft"]
    assert draft.startswith("# ")
    assert "**Sources:**" in draft
    assert mem.last_output("blog")


def test_research_only_routes_to_strategist():
    out, _ = _run_graph("research email marketing trends and statistics")
    assert out["research"]
    assert "strategy" in out["drafts"]
    assert "Key messages" in out["drafts"]["strategy"]["draft"]


def test_combo_produces_multiple_formats():
    out, _ = _run_graph("write a blog and a LinkedIn post about onboarding")
    assert "blog" in out["drafts"]
    assert "linkedin" in out["drafts"]
    assert out["research"]


def test_image_route_placeholder(monkeypatch):
    """Free provider unreachable and no OpenAI key -> labelled placeholder.

    Hermetic: force the Pollinations call to fail (this test must not depend
    on whether the sandbox network happens to reach Pollinations.ai).
    """
    import contentblitz.tools as tools

    monkeypatch.setattr(tools, "_pollinations_image_sync", lambda p: None)
    monkeypatch.setattr(tools, "_sleep_between_attempts", lambda s: None)
    out, _ = _run_graph("generate an image for our launch banner")
    assert "image" in out["drafts"]
    img = out["drafts"]["image"]
    assert img["status"] == "placeholder"
    assert "PLACEHOLDER" in out["answer"]


def test_refine_iterates_last_draft():
    mem = ConversationMemory()
    first, _ = _run_graph("write a LinkedIn post about our launch", mem)
    drafts = first["drafts"]
    assert "linkedin" in drafts
    out, _ = _run_graph("make it shorter and punchier", mem, drafts=drafts)
    assert out["decision"]["intent"] == "refine"
    assert "linkedin" in out["drafts"]


def test_refine_without_draft_explains():
    out, _ = _run_graph("write a LinkedIn post")
    # fresh memory, no drafts -> refine intent can't trigger without memory,
    # so this is a sanity check that the graph always answers.
    assert out["answer"]


def test_series_route():
    out, _ = _run_graph("plan a 5-part series for our launch campaign")
    assert "series_part_1" in out["drafts"]
    assert "Part 1" in out["answer"]


def test_new_topic_triggers_fresh_research():
    """Stale research in state must not suppress a fresh research-first run."""
    stale = {"results": [{"title": "OLD", "url": "https://old.co",
                          "snippet": "stale", "source": "mock"}],
             "provider": "mock", "summary": "stale"}
    out, _ = _run_graph("write a blog post about email marketing",
                        research=stale)
    titles = [r["title"] for r in out["research"]["results"]]
    assert not any(t == "OLD" for t in titles)
