"""The six specialized ContentBlitz agents.

1. Query Handler — intelligent LLM-based router (also the LangGraph router
   node in graph.py). Makes dynamic routing decisions as structured JSON:
   intent, whether research is needed, target formats. Handles BOTH initial
   routing AND post-research routing. Keyword fallback when offline.
2. Deep Research Agent — comprehensive web research with citations.
3. SEO Blog Writer — search-optimized long-form content.
4. LinkedIn Post Writer — professional social content.
5. Image Generation Agent — Pollinations (free/keyless) -> DALL-E 3 ->
   DALL-E 2 -> labelled placeholder.
6. Content Strategist — organizes research into readable content plans.

Each worker is an async callable taking (brief, research/context, memory)
and returning a result dict. Without an OpenAI key they degrade to
deterministic template writers labelled as offline.
"""
from __future__ import annotations

import asyncio
import json
import re

from contentblitz.brand import voice_guidance
from contentblitz.config import settings
from contentblitz.quality import seo_optimize, validate_content
from contentblitz.tools import generate_image, optimize_image_prompt, web_research

OFFLINE_TAG = "\n\n_(Drafted in offline mode — set OPENAI_API_KEY for AI-polished copy.)_"


def _apply_offline_refinement(draft: str, feedback: str | None) -> str:
    """Deterministic offline refinement of an existing draft.

    Mirrors what the LLM path does (edit the draft, don't start over) so
    hermetic tests exercise real multi-turn behavior. Feedback with no
    recognized directive is returned unchanged — the draft is still the
    prior one, never a fresh topic.
    """
    if not feedback or not draft:
        return draft
    f = feedback.lower()
    if "shorter" in f:
        paras = [p for p in draft.split("\n\n") if p.strip()]
        keep = max(2, len(paras) * 2 // 3)
        return "\n\n".join(paras[:keep])
    if "longer" in f:
        return (draft
                + "\n\n## One more angle\n\nGo one level deeper on the core "
                  "idea above before acting on it.")
    if "punchier" in f or "hook" in f:
        return "🔥 **Stop scrolling — read this first.**\n\n" + draft
    if "formal" in f or "tone down" in f:
        return draft.replace("🔥", "").replace("!", ".")
    return draft


# ---------------------------------------------------------------------------
# LLM plumbing (mirrors the Finnie pattern)
# ---------------------------------------------------------------------------
def _llm():
    if not settings.llm_available:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.LLM_MODEL, temperature=0.7)


async def _ask_llm(system: str, user: str) -> str | None:
    llm = _llm()
    if llm is None:
        return None
    try:
        resp = await asyncio.to_thread(
            llm.invoke,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
        )
        return resp.content
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Query Handler — intelligent router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are ContentBlitz's Query Handler, an intelligent content-
marketing router. Analyze the user's request (and conversation history) and
reply with ONLY a JSON object, no other text:

{
  "intent": one of ["research", "blog", "linkedin", "image", "strategy",
                   "refine", "series", "chat", "combo"],
  "needs_research": true/false,
  "formats": subset of ["blog", "linkedin", "image", "strategy"],
  "target_kind": the format to produce when intent is "refine" and the user
                 names one (e.g. "turn that into a LinkedIn post" -> "linkedin"),
                 else null,
  "confidence": 0.0-1.0,
  "reasoning": "one short sentence"
}

Intent guide:
- research: user wants facts, trends, competitor info about a topic
- blog: write/publish a long-form SEO article
- linkedin: write a LinkedIn post
- image: create/generate a visual
- strategy: content plan, calendar, key messages, repurposing plan
- refine: user gives feedback on a previous draft ("shorter", "punchier",
  "add stats") OR asks to repurpose an existing draft into another format
  ("turn that into a LinkedIn post", "make this a blog") — never treat the
  follow-up text itself as a new topic
- series: multi-part campaign ("5-part series", "week of posts")
- combo: explicitly wants several formats ("blog and LinkedIn post")
- chat: greetings, questions about capabilities, anything else

Set needs_research=true when the request needs current facts, stats, or
examples (research-first workflow); false for pure rewrites of existing
drafts or chit-chat. For refine/series/chat, formats may be []."""

# Deterministic keyword fallback when no LLM key is configured. The LLM
# router above is the real decision maker; this keeps the app fully usable
# offline (e.g. Streamlit Cloud without secrets).
_FORMAT_KEYWORDS = {
    "blog": ["blog", "article", "seo", "long-form", "long form"],
    "linkedin": ["linkedin", "social post", "social media"],
    "image": ["image", "picture", "visual", "graphic", "illustration",
              "banner", "thumbnail", "dall-e", "generate art"],
    "strategy": ["strategy", "content plan"],
}
_SERIES_KEYWORDS = ["series", "campaign", "5-part", "multi-part", "week of",
                    "content calendar"]
_RESEARCH_KEYWORDS = ["research", "trends", "statistics", "stats",
                      "competitor", "landscape", "find out", "latest"]
_STRATEGY_KEYWORDS = ["strategy", "key messages", "positioning", "editorial"]
_REFINE_KEYWORDS = ["shorter", "longer", "punchier", "rewrite", "revise",
                    "make it", "tone down", "more formal", "add stats",
                    "simplify", "improve", "better hook"]
# Follow-ups that repurpose an existing draft into another format
# ("turn that into a LinkedIn post") — refinements, not new topics.
_REPURPOSE_LEADS = ["turn that into", "turn this into", "convert that",
                    "convert this", "make that into", "make this into",
                    "make that a ", "make this a ", "repurpose",
                    "into a linkedin post", "as a linkedin post",
                    "into a blog post", "as a blog post"]


def route_query_sync(user_input: str, memory=None) -> dict:
    """Deterministic keyword fallback — also used in tests."""
    q = user_input.lower()
    has_draft = memory is not None and bool(memory.last_outputs)

    # Repurpose: "turn that into a LinkedIn post" edits the existing draft
    # into another format — it must not become a new topic.
    if has_draft and any(lead in q for lead in _REPURPOSE_LEADS):
        kinds = list(memory.last_outputs)
        target = next((fmt for fmt, kws in _FORMAT_KEYWORDS.items()
                       if any(k in q for k in kws)), kinds[-1])
        return {"intent": "refine", "needs_research": False,
                "formats": [target], "target_kind": target,
                "confidence": 0.8,
                "reasoning": "Repurpose follow-up on existing draft (offline mode)."}

    # Refinement only makes sense when a draft exists.
    if has_draft and any(k in q for k in _REFINE_KEYWORDS):
        kinds = list(memory.last_outputs)
        return {"intent": "refine", "needs_research": False,
                "formats": kinds[-1:], "target_kind": kinds[-1],
                "confidence": 0.8,
                "reasoning": "Refinement keywords + existing draft in memory."}

    formats = [fmt for fmt, kws in _FORMAT_KEYWORDS.items()
               if any(k in q for k in kws)]
    wants_research = any(k in q for k in _RESEARCH_KEYWORDS)

    if any(k in q for k in _SERIES_KEYWORDS):
        return {"intent": "series", "needs_research": True, "formats": [],
                "confidence": 0.7,
                "reasoning": "Series/campaign keywords (offline mode)."}

    if wants_research:
        # "research X and write a blog" -> research-first blog, not bare research.
        if len(formats) >= 2:
            intent, out_formats = "combo", formats
        elif len(formats) == 1:
            intent, out_formats = formats[0], formats
        else:
            intent, out_formats = "research", []
        return {"intent": intent, "needs_research": True,
                "formats": out_formats, "confidence": 0.75,
                "reasoning": "Research keywords (offline mode)."}

    if any(k in q for k in _STRATEGY_KEYWORDS) or "strategy" in formats:
        return {"intent": "strategy", "needs_research": False,
                "formats": ["strategy"], "confidence": 0.7,
                "reasoning": "Strategy keywords (offline mode)."}

    if len(formats) >= 2:
        return {"intent": "combo", "needs_research": True,
                "formats": formats, "confidence": 0.7,
                "reasoning": "Multiple formats requested (offline mode)."}
    if len(formats) == 1:
        intent = formats[0]
        return {"intent": intent,
                "needs_research": intent == "blog",
                "formats": formats, "confidence": 0.65,
                "reasoning": f"Single format keyword: {intent} (offline mode)."}

    return {"intent": "chat", "needs_research": False, "formats": [],
            "confidence": 0.6,
            "reasoning": "No routing keywords matched (offline mode)."}


async def route_query(user_input: str, memory=None) -> dict:
    """Intelligent LLM routing with deterministic fallback.

    This is the Query Handler agent: it makes dynamic, context-aware
    decisions — never a bare if/else chain — and is invoked both for the
    initial request and for post-research routing.
    """
    history = memory.history_text() if memory is not None else ""
    if settings.llm_available:
        prompt = (f"Conversation history:\n{history}\n\nUser request: {user_input}")
        raw = await _ask_llm(ROUTER_SYSTEM, prompt)
        if raw:
            try:
                cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(),
                                 flags=re.M)
                decision = json.loads(cleaned.strip())
                if decision.get("intent") in {
                        "research", "blog", "linkedin", "image", "strategy",
                        "refine", "series", "chat", "combo"}:
                    decision.setdefault("needs_research", False)
                    decision.setdefault("formats", [])
                    decision.setdefault("confidence", 0.8)
                    decision.setdefault("reasoning", "LLM routing decision.")
                    return decision
            except Exception:
                pass
    return route_query_sync(user_input, memory)


# ---------------------------------------------------------------------------
# 2. Deep Research Agent
# ---------------------------------------------------------------------------
async def deep_research_agent(brief: dict, memory=None) -> dict:
    """Comprehensive web research with citations.

    Fans out across the topic + 2-3 angled sub-queries, then compiles a
    cited research pack the writers can ground on.
    """
    topic = brief.get("topic", "")
    audience = brief.get("audience", "")
    angles = [
        topic,
        f"{topic} trends statistics",
        f"{topic} best practices {audience}".strip(),
    ]
    packs = []
    for angle in angles:
        if angle.strip():
            packs.append(await web_research(angle))
    seen: list[dict] = []
    urls: set[str] = set()
    for pack in packs:
        for r in pack["results"]:
            if r["url"] and r["url"] not in urls:
                urls.add(r["url"])
                seen.append(r)
    provider = packs[0]["provider"] if packs else "mock"
    summary = await _ask_llm(
        "You are a research analyst. Summarize the findings below into 5-8 "
        "tight bullet points a content writer can use. Keep facts attributable.",
        "\n\n".join(f"- {r['title']}: {r['snippet']}" for r in seen[:10]),
    )
    if not summary:
        summary = "\n".join(
            f"- **{r['title']}** — {r['snippet'][:160]}" for r in seen[:8]
        ) + OFFLINE_TAG
    pack = {
        "topic": topic,
        "provider": provider,
        "results": seen[:12],
        "summary": summary,
        "citation_count": len(seen),
    }
    if memory is not None:
        memory.remember_research(pack["results"])
    return pack


def format_citations(research: dict) -> str:
    lines = ["\n\n**Sources:**"]
    for i, r in enumerate(research.get("results", [])[:8], 1):
        label = {"tavily": "live", "serpapi": "live", "mock": "demo"}.get(
            r.get("source"), "demo")
        lines.append(f"{i}. [{r['title']}]({r['url']}) _({label})_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. SEO Blog Writer
# ---------------------------------------------------------------------------
async def seo_blog_writer(brief: dict, research: dict | None = None,
                          memory=None, feedback: str | None = None,
                          prior_draft: str | None = None) -> dict:
    """Search-optimized long-form article, run through the SEO engine.

    When ``prior_draft`` is given (a follow-up refinement), the writer
    edits that draft in place: both the draft and the feedback reach the
    model, so "make it punchier" revises instead of restarting.
    """
    topic = brief.get("topic", "Untitled")
    audience = brief.get("audience", "general readers")
    voice = voice_guidance(brief.get("brand_voice"))
    research_block = ""
    if research:
        research_block = research.get("summary", "")

    if prior_draft:
        system = (
            "You are an expert SEO content writer. Revise the CURRENT DRAFT "
            f"below according to the revision instructions. {voice} Keep the "
            "topic, structure, and research grounding; change only what the "
            "instructions ask for. Return the full revised article. "
            "Markdown only."
        )
        user = (f"Topic: {topic}\nAudience: {audience}\n\n"
                f"CURRENT DRAFT (revise this, do not start over):\n"
                f"{prior_draft}\n\n"
                f"Revision instructions: {feedback or 'improve it'}\n\n"
                "Return the full revised article now.")
    else:
        system = (
            "You are an expert SEO content writer. Write a complete, original "
            f"long-form blog article. {voice} Structure: compelling H1, intro "
            "with hook, 4-6 H2 sections with H3 subsections, concrete examples "
            "and numbers from the research, FAQ section, conclusion with CTA. "
            "Weave the research findings in naturally. Markdown only."
        )
        user = (f"Topic: {topic}\nAudience: {audience}\n\nResearch:\n{research_block}\n"
                + (f"\nRevision feedback (apply it): {feedback}\n" if feedback else "")
                + "\nWrite the full article now.")
    draft = await _ask_llm(system, user)
    offline = not draft
    if offline:
        draft = (_apply_offline_refinement(prior_draft, feedback)
                 if prior_draft else _offline_blog(brief, research))

    seo = seo_optimize(draft, brief, (research or {}).get("results", []))
    checked = validate_content(seo["optimized"], "blog", brief.get("brand_voice"))
    final = checked["enhanced"] + format_citations(research or {"results": []})
    if memory is not None:
        memory.remember_output("blog", final)
        memory.remember_topic(topic)
    return {
        "kind": "blog",
        "draft": final,
        "seo": {k: v for k, v in seo.items() if k != "optimized"},
        "quality": checked["score"],
        "voice_check": checked["voice_check"],
        "issues_fixed": checked["issues"],
        "offline": offline,
    }


def _offline_blog(brief: dict, research: dict | None) -> str:
    topic = brief.get("topic", "Untitled")
    audience = brief.get("audience", "general readers")
    points = []
    if research:
        for r in research.get("results", [])[:5]:
            points.append(f"### {r['title']}\n\n{r['snippet']}\n")
    body = "\n".join(points) or "Key points coming soon."
    return (
        f"# {topic}: A Practical Guide\n\n"
        f"*{audience.capitalize()} — here's what you need to know about {topic}.*\n\n"
        f"## Why {topic} matters now\n\nEvery {audience} team is asking how {topic} "
        f"fits their roadmap. This guide distills the essentials.\n\n"
        f"## Key insights\n\n{body}\n"
        f"## Getting started\n\n1. Define your goal for {topic}.\n"
        f"2. Start small and measure.\n3. Iterate based on feedback.\n\n"
        f"## FAQ\n\n**What is {topic}?**\n\nSee the insights above for a working definition.\n\n"
        f"## Conclusion\n\n{topic} rewards teams that start now. Pick one action "
        f"from this guide and ship it this week."
        + OFFLINE_TAG
    )


# ---------------------------------------------------------------------------
# 4. LinkedIn Post Writer
# ---------------------------------------------------------------------------
async def linkedin_writer(brief: dict, research: dict | None = None,
                          memory=None, feedback: str | None = None,
                          prior_draft: str | None = None) -> dict:
    """LinkedIn ghostwriter.

    When ``prior_draft`` is given (e.g. "turn that into a LinkedIn post"
    or "make it punchier"), the draft reaches the model alongside the
    feedback, so follow-ups edit the existing draft instead of treating
    the follow-up text as a new topic.
    """
    topic = brief.get("topic", "Untitled")
    voice = voice_guidance(brief.get("brand_voice"))
    research_block = (research or {}).get("summary", "")

    if prior_draft:
        system = (
            "You are a LinkedIn ghostwriter for executives. Revise or "
            f"repurpose the CURRENT DRAFT below per the instructions. {voice} "
            "Rules for the result: scroll-stopping first line, short lines "
            "with breathing room, keep the draft's substance and any concrete "
            "insight or number, end with a question CTA, 3-5 relevant "
            "hashtags. No clickbait, no emojis spam. Return the full post."
        )
        user = (f"Topic: {topic}\nResearch:\n{research_block}\n\n"
                f"CURRENT DRAFT (revise/repurpose this, do not start over):\n"
                f"{prior_draft}\n\n"
                f"Instructions: {feedback or 'improve it'}\n\n"
                "Return the full LinkedIn post now.")
    else:
        system = (
            "You are a LinkedIn ghostwriter for executives. Write ONE LinkedIn "
            f"post (120-220 words). {voice} Rules: scroll-stopping first line, "
            "short lines with breathing room, one concrete insight or number "
            "from the research, end with a question CTA, 3-5 relevant hashtags. "
            "No clickbait, no emojis spam."
        )
        user = (f"Topic: {topic}\nResearch:\n{research_block}\n"
                + (f"\nRevision feedback (apply it): {feedback}\n" if feedback else "")
                + "\nWrite the post now.")
    draft = await _ask_llm(system, user)
    if not draft:
        draft = _offline_linkedin(brief, research, feedback, prior_draft)
    checked = validate_content(draft, "linkedin", brief.get("brand_voice"))
    if memory is not None:
        memory.remember_output("linkedin", checked["enhanced"])
        memory.remember_topic(topic)
    return {
        "kind": "linkedin",
        "draft": checked["enhanced"],
        "quality": checked["score"],
        "voice_check": checked["voice_check"],
        "issues_fixed": checked["issues"],
    }


def _is_repurpose_feedback(feedback: str | None) -> bool:
    """Does the feedback ask to convert the draft into another format?"""
    if not feedback:
        return False
    f = feedback.lower()
    return any(h in f for h in (
        "turn that into", "turn this into", "convert that", "convert this",
        "repurpose", "into a linkedin post", "into a blog post",
        "as a linkedin post", "as a blog post"))


def _offline_linkedin(brief: dict, research: dict | None,
                      feedback: str | None = None,
                      prior_draft: str | None = None) -> str:
    topic = brief.get("topic", "Untitled")
    if prior_draft and not _is_repurpose_feedback(feedback):
        # Same-format refinement: edit the draft in place, don't restart.
        return _apply_offline_refinement(prior_draft, feedback)
    if prior_draft:
        # Honest repurpose: distill the actual prior draft, don't invent.
        src = " ".join(prior_draft.split())
        excerpt = src[:300].rsplit(" ", 1)[0] if len(src) > 300 else src
        tag = (topic.split() or ["Untitled"])[0].title()
        post = (
            f"Distilled from my latest piece on {topic}:\n\n"
            f"{excerpt}…\n\n"
            f"What's your #1 lesson with {topic}?\n\n"
            f"#{tag} #Marketing #Growth"
            + OFFLINE_TAG
        )
        return _apply_offline_refinement(post, feedback)
    stat = ""
    if research and research.get("results"):
        stat = research["results"][0]["snippet"][:140]
    return (
        f"I used to overcomplicate {topic}.\n\n"
        f"Then I stripped it to what actually moves the needle:\n\n"
        f"→ Start with one clear goal\n→ Measure what matters\n→ Iterate weekly\n\n"
        + (f"One data point worth knowing: {stat}\n\n" if stat else "")
        + f"What's your #1 lesson with {topic}?\n\n"
        f"#{topic.split()[0].title()} #Marketing #Growth"
        + OFFLINE_TAG
    )


# ---------------------------------------------------------------------------
# 5. Image Generation Agent
# ---------------------------------------------------------------------------
async def image_agent(brief: dict, memory=None) -> dict:
    """Prompt optimization + Pollinations (free/keyless) -> DALL-E 3 ->
    DALL-E 2 -> labelled placeholder."""
    prompt = optimize_image_prompt(brief)
    # Optional LLM polish of the visual prompt.
    polished = await _ask_llm(
        "You are an art director. Improve this image prompt for a marketing "
        "visual: keep it under 80 words, vivid, no text in image.",
        prompt,
    )
    final_prompt = (polished or "").strip() or prompt
    result = await generate_image(final_prompt)
    if memory is not None:
        memory.remember_output(
            "image", result.get("image_url") or result.get("note", ""))
    return {"kind": "image", **result}


# ---------------------------------------------------------------------------
# 6. Content Strategist
# ---------------------------------------------------------------------------
async def content_strategist(brief: dict, research: dict | None = None,
                             memory=None, feedback: str | None = None,
                             prior_draft: str | None = None) -> dict:
    """Organize research into a readable strategy: messages, outline, plan.

    ``prior_draft`` lets follow-ups revise the strategy in place rather
    than regenerating it from scratch.
    """
    topic = brief.get("topic", "Untitled")
    audience = brief.get("audience", "general readers")
    voice = voice_guidance(brief.get("brand_voice"))
    research_block = (research or {}).get("summary", "")

    if prior_draft:
        system = (
            "You are a content strategist. Revise the CURRENT STRATEGY DRAFT "
            f"below per the instructions. {voice} Keep what works; change "
            "only what the instructions ask for. Return the full revised "
            "strategy doc. Markdown, scannable."
        )
        user = (f"Topic: {topic}\nAudience: {audience}\n\n"
                f"CURRENT DRAFT (revise this, do not start over):\n"
                f"{prior_draft}\n\n"
                f"Instructions: {feedback or 'improve it'}\n\n"
                "Return the full revised strategy doc now.")
    else:
        system = (
            "You are a content strategist. Turn the research into a crisp content "
            f"strategy doc. {voice} Include: 1) positioning statement, 2) 3-5 key "
            "messages, 3) recommended content outline, 4) repurposing plan "
            "(1 blog -> LinkedIn posts -> visuals), 5) distribution notes. "
            "Markdown, scannable."
        )
        user = (f"Topic: {topic}\nAudience: {audience}\n\nResearch:\n{research_block}\n"
                + (f"\nRevision feedback (apply it): {feedback}\n" if feedback else "")
                + "\nWrite the strategy doc now.")
    draft = await _ask_llm(system, user)
    if not draft:
        draft = (_apply_offline_refinement(prior_draft, feedback)
                 if prior_draft else _offline_strategy(brief, research))
    checked = validate_content(draft, "strategy", brief.get("brand_voice"))
    if memory is not None:
        memory.remember_output("strategy", checked["enhanced"])
        memory.remember_topic(topic)
    return {
        "kind": "strategy",
        "draft": checked["enhanced"],
        "quality": checked["score"],
        "voice_check": checked["voice_check"],
        "issues_fixed": checked["issues"],
    }


def _offline_strategy(brief: dict, research: dict | None) -> str:
    topic = brief.get("topic", "Untitled")
    audience = brief.get("audience", "general readers")
    return (
        f"# Content Strategy: {topic}\n\n"
        f"## Positioning\n\n{topic} — explained for {audience}, without the fluff.\n\n"
        f"## Key messages\n\n1. {topic} matters now.\n"
        f"2. Start small, measure, iterate.\n3. Practical beats theoretical.\n\n"
        f"## Recommended outline\n\n- Hook: the problem {audience} feel\n"
        f"- Insights from research\n- Actionable next steps\n\n"
        f"## Repurposing plan\n\n1. Long-form blog (SEO anchor)\n"
        f"2. 3 LinkedIn posts (one insight each)\n3. 1 hero visual for social\n\n"
        f"## Distribution\n\nPublish blog first, drip social over the following week."
        + OFFLINE_TAG
    )


AGENTS = {
    "query_handler": route_query,          # router (see graph.py)
    "deep_research": deep_research_agent,
    "seo_blog": seo_blog_writer,
    "linkedin": linkedin_writer,
    "image": image_agent,
    "strategist": content_strategist,
}
