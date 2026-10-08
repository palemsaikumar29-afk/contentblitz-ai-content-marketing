"""Regression tests for grader feedback: conversational memory.

Covers the production lever called out in grading — follow-ups like
"turn that into a LinkedIn post" or "make it punchier" must carry the
real topic AND the last draft forward, editing the existing draft
instead of treating the follow-up text as a new topic.
"""
import asyncio

from contentblitz.agents import (
    content_strategist,
    linkedin_writer,
    route_query_sync,
    seo_blog_writer,
)
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


# -- memory ---------------------------------------------------------------
def test_remember_topic_and_last_draft():
    mem = ConversationMemory()
    assert mem.last_topic == ""
    assert mem.last_draft() == (None, None)
    mem.remember_topic("email marketing")
    mem.remember_output("blog", "blog draft")
    mem.remember_output("linkedin", "linkedin draft")
    assert mem.last_topic == "email marketing"
    assert mem.last_draft() == ("linkedin", "linkedin draft")
    assert mem.last_draft("blog") == ("blog", "blog draft")
    assert mem.last_draft("image") == (None, None)


def test_remember_topic_ignores_blank():
    mem = ConversationMemory()
    mem.remember_topic("  ")
    mem.remember_topic(None)
    assert mem.last_topic == ""


def test_clear_resets_topic():
    mem = ConversationMemory()
    mem.remember_topic("x")
    mem.clear()
    assert mem.last_topic == ""


def test_writers_anchor_topic_in_memory():
    mem = ConversationMemory()
    asyncio.run(seo_blog_writer(BRIEF, None, mem))
    assert mem.last_topic == "email marketing"
    asyncio.run(linkedin_writer(BRIEF, None, mem))
    assert mem.last_topic == "email marketing"


# -- router ---------------------------------------------------------------
def test_sync_repurpose_routes_to_refine_with_target():
    mem = ConversationMemory()
    mem.remember_output("blog", "some blog draft")
    d = route_query_sync("turn that into a LinkedIn post", mem)
    assert d["intent"] == "refine"
    assert d["target_kind"] == "linkedin"
    assert d["needs_research"] is False


def test_sync_repurpose_without_draft_is_not_refine():
    mem = ConversationMemory()
    d = route_query_sync("turn that into a LinkedIn post", mem)
    assert d["intent"] != "refine"  # no draft exists -> fresh write


def test_sync_repurpose_defaults_to_last_draft_kind():
    mem = ConversationMemory()
    mem.remember_output("strategy", "some strategy")
    d = route_query_sync("turn this into something visual", mem)
    assert d["intent"] == "refine"
    assert d["target_kind"] in {"strategy", "image"}


# -- writers carry the draft to the model ---------------------------------
def test_linkedin_writer_sends_prior_draft_to_llm(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    seen = {}

    async def fake_ask(system, user):
        seen["user"] = user
        return "Revised post.\n\nAgree? #Growth"

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    asyncio.run(linkedin_writer(BRIEF, None, None,
                                feedback="make it punchier",
                                prior_draft="ORIGINAL DRAFT TEXT"))
    assert "ORIGINAL DRAFT TEXT" in seen["user"]
    assert "make it punchier" in seen["user"]


def test_offline_refinement_edits_prior_draft():
    mem = ConversationMemory()
    original = asyncio.run(linkedin_writer(BRIEF, None, mem))["draft"]
    refined = asyncio.run(
        linkedin_writer(BRIEF, None, mem, feedback="make it punchier",
                        prior_draft=original))["draft"]
    assert refined != original
    assert refined.startswith("🔥")  # punchier hook applied
    assert "overcomplicate" in refined  # original substance retained
    # The follow-up text never becomes the topic.
    assert "make it punchier" not in refined.replace("🔥", "")


def test_offline_shorter_actually_shortens():
    original = "para one.\n\npara two.\n\npara three.\n\npara four."
    out = asyncio.run(linkedin_writer(BRIEF, None, None,
                                      feedback="make it shorter",
                                      prior_draft=original))["draft"]
    assert len(out) < len(original)


# -- graph end to end ------------------------------------------------------
def test_repurpose_keeps_real_topic():
    """'turn that into a LinkedIn post' after a blog -> LinkedIn post
    about the ORIGINAL topic, not about the follow-up text."""
    mem = ConversationMemory()
    first, _ = _run_graph("write a blog post about email marketing", mem)
    assert "blog" in first["drafts"]
    assert mem.last_topic == "email marketing"

    out, _ = _run_graph("turn that into a LinkedIn post", mem,
                        drafts=first["drafts"])
    assert out["decision"]["intent"] == "refine"
    assert out["decision"].get("target_kind") == "linkedin"
    assert "linkedin" in out["drafts"]
    post = out["drafts"]["linkedin"]["draft"]
    assert "email marketing" in post  # real topic carried forward
    assert mem.last_topic == "email marketing"  # topic persists


def test_refine_edits_last_draft_not_restart():
    mem = ConversationMemory()
    first, _ = _run_graph("write a LinkedIn post about our launch", mem)
    original = first["drafts"]["linkedin"]["draft"]

    out, _ = _run_graph("make it punchier", mem, drafts=first["drafts"])
    assert out["decision"]["intent"] == "refine"
    refined = out["drafts"]["linkedin"]["draft"]
    assert refined != original
    assert "email marketing" in refined  # still about the original topic
    assert refined.startswith("🔥")  # punchier hook applied in place
    assert mem.last_topic != "make it punchier"


def test_refine_without_any_draft_explains():
    out, _ = _run_graph("make it punchier")
    assert out["answer"]


# -- offline refinement edge branches --------------------------------------
def test_offline_refinement_no_feedback_returns_draft():
    from contentblitz.agents import _apply_offline_refinement
    assert _apply_offline_refinement("draft text", None) == "draft text"
    assert _apply_offline_refinement("draft text", "") == "draft text"


def test_offline_refinement_unknown_directive_keeps_draft():
    from contentblitz.agents import _apply_offline_refinement
    out = _apply_offline_refinement("draft text", "add more cowbell")
    assert out == "draft text"  # honest: no invented transform


def test_offline_refinement_formal_tones_down():
    from contentblitz.agents import _apply_offline_refinement
    out = _apply_offline_refinement("Wow! Amazing!", "make it more formal")
    assert "!" not in out
    assert "🔥" not in out


def test_blog_writer_sends_prior_draft_to_llm(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    seen = {}

    async def fake_ask(system, user):
        seen["system"] = system
        seen["user"] = user
        return "# Revised\n\nBody."

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    asyncio.run(seo_blog_writer(BRIEF, None, None,
                                feedback="add stats",
                                prior_draft="OLD BLOG DRAFT"))
    assert "OLD BLOG DRAFT" in seen["user"]
    assert "add stats" in seen["user"]
    assert "Revise" in seen["system"]


def test_strategist_writer_sends_prior_draft_to_llm(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg

    seen = {}

    async def fake_ask(system, user):
        seen["user"] = user
        return "Revised strategy."

    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    asyncio.run(content_strategist(BRIEF, None, None,
                                   feedback="sharper messages",
                                   prior_draft="OLD STRATEGY DRAFT"))
    assert "OLD STRATEGY DRAFT" in seen["user"]
    assert "sharper messages" in seen["user"]
