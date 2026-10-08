"""LangGraph orchestration for ContentBlitz.

The Query Handler is the router node. It is invoked TWICE in the
research-first workflow:

1. **Initial routing** — classifies the request; decides whether deep
   research is needed and which formats to produce.
2. **Post-research routing** — after the Deep Research Agent finishes, the
   flow returns to the Query Handler, which now sees the research pack in
   state and routes to the SEO Blog Writer / LinkedIn Writer / Content
   Strategist / a combination of them.

The router is an intelligent LLM-based decision maker (agents.route_query),
never a bare if/else chain — the keyword fallback only engages offline.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from contentblitz import agents
from contentblitz.agents import (
    content_strategist,
    deep_research_agent,
    image_agent,
    linkedin_writer,
    route_query,
    seo_blog_writer,
)


class BlitzState(TypedDict, total=False):
    user_input: str
    brief: dict
    decision: dict
    research: dict
    pending: list          # formats queued for post-research routing
    drafts: dict           # kind -> agent result dict
    feedback: str | None
    answer: str            # final user-facing message
    _researched: bool      # internal: research-first leg requested/completed


_FORMAT_TO_NODE = {
    "blog": "seo_blog",
    "linkedin": "linkedin",
    "image": "image",
    "strategy": "strategist",
}


def build_graph(memory=None):
    """Build the ContentBlitz graph. ``memory`` is session conversation memory."""
    builder = StateGraph(BlitzState)

    # -- router ---------------------------------------------------------
    async def query_handler(state: BlitzState) -> BlitzState:
        decision = await route_query(state.get("user_input", ""), memory)
        state = {**state, "decision": decision}
        if state.get("_researched"):
            # Post-research routing: the research pack just completed.
            # Consume the queued formats (or fall back to the fresh decision).
            pending = state.get("pending") or decision.get("formats") or []
            if not pending and decision.get("intent") == "research":
                pending = ["strategy"]  # organize findings into readable content
            state["pending"] = pending
            state["_researched"] = False
        elif decision.get("needs_research"):
            # Fresh research-first workflow (overwrites any stale research).
            # Bare research requests default to a strategy write-up of findings.
            pending = decision.get("formats") or []
            if not pending and decision.get("intent") == "research":
                pending = ["strategy"]
            state["pending"] = pending
            state["_researched"] = True
        return state

    def _route(state: BlitzState) -> str:
        # Fresh research needed -> research-first workflow.
        if state.get("_researched"):
            return "deep_research"
        # Post-research: dispatch the queued formats.
        pending = state.get("pending") or []
        if state.get("research") and pending:
            if len(pending) > 1:
                return "combo"
            return _FORMAT_TO_NODE.get(pending[0], "strategist")
        decision = state.get("decision") or {}
        intent = decision.get("intent", "chat")
        if intent == "refine":
            return "refine"
        if intent == "series":
            return "series"
        if intent == "chat":
            return "chat"
        formats = decision.get("formats") or []
        if len(formats) > 1:
            return "combo"
        if formats:
            return _FORMAT_TO_NODE.get(formats[0], "chat")
        return {"research": "deep_research", "blog": "seo_blog",
                "linkedin": "linkedin", "image": "image",
                "strategy": "strategist"}.get(intent, "chat")

    # -- worker nodes ----------------------------------------------------
    async def deep_research(state: BlitzState) -> BlitzState:
        research = await deep_research_agent(state.get("brief") or {}, memory)
        return {**state, "research": research}

    async def seo_blog(state: BlitzState) -> BlitzState:
        result = await seo_blog_writer(
            state.get("brief") or {}, state.get("research"), memory)
        drafts = {**(state.get("drafts") or {}), "blog": result}
        return {**state, "drafts": drafts,
                "answer": result["draft"]}

    async def linkedin(state: BlitzState) -> BlitzState:
        result = await linkedin_writer(
            state.get("brief") or {}, state.get("research"), memory)
        drafts = {**(state.get("drafts") or {}), "linkedin": result}
        return {**state, "drafts": drafts,
                "answer": result["draft"]}

    async def image(state: BlitzState) -> BlitzState:
        result = await image_agent(state.get("brief") or {}, memory)
        drafts = {**(state.get("drafts") or {}), "image": result}
        return {**state, "drafts": drafts,
                "answer": _image_answer(result)}

    async def strategist(state: BlitzState) -> BlitzState:
        result = await content_strategist(
            state.get("brief") or {}, state.get("research"), memory)
        drafts = {**(state.get("drafts") or {}), "strategy": result}
        return {**state, "drafts": drafts,
                "answer": result["draft"]}

    async def combo(state: BlitzState) -> BlitzState:
        """Combination node: run several writers for one brief."""
        pending = state.get("pending") or (state.get("decision") or {}).get(
            "formats") or ["blog", "linkedin"]
        brief = state.get("brief") or {}
        research = state.get("research")
        drafts = dict(state.get("drafts") or {})
        runners = {"blog": seo_blog_writer, "linkedin": linkedin_writer,
                   "image": image_agent, "strategy": content_strategist}
        for fmt in pending:
            fn = runners.get(fmt)
            if not fn:
                continue
            if fmt == "image":
                result = await fn(brief, memory)
            else:
                result = await fn(brief, research, memory)
            drafts[fmt] = result
        parts = [f"## {k.title()}\n\n{d['draft']}" for k, d in drafts.items()
                 if d.get("draft")]
        return {**state, "drafts": drafts, "pending": [],
                "answer": "\n\n---\n\n".join(parts)}

    async def refine(state: BlitzState) -> BlitzState:
        """Multi-turn iterative refinement of the last draft.

        Carries the REAL topic (memory.last_topic) and the actual prior
        draft into the writer, so follow-ups like "make it punchier" or
        "turn that into a LinkedIn post" edit the existing draft instead
        of treating the follow-up text as a new topic.
        """
        feedback = state.get("feedback") or state.get("user_input", "")
        decision = state.get("decision") or {}
        drafts = dict(state.get("drafts") or {})
        runners = {"blog": seo_blog_writer, "linkedin": linkedin_writer,
                   "strategy": content_strategist}

        # Resolve the prior draft: memory is the session truth, state
        # drafts the fallback (e.g. restored UI session state).
        target_kind = decision.get("target_kind")
        prior_kind, prior_draft = None, None
        if memory is not None:
            prior_kind, prior_draft = memory.last_draft(target_kind)
        if prior_draft is None and drafts:
            prior_kind = (target_kind if target_kind in drafts
                          else next(reversed(drafts.keys())))
            prior_draft = (drafts.get(prior_kind) or {}).get("draft")
        kind = target_kind or prior_kind
        if not prior_draft or kind not in {**runners, "image": image_agent}:
            return {**state, "answer": "There's no draft to refine yet — ask me to write something first."}

        brief = dict(state.get("brief") or {})
        if memory is not None and memory.last_topic:
            # The real topic — never the follow-up text ("make it punchier").
            brief["topic"] = memory.last_topic
        if kind == "image":
            result = await image_agent(brief, memory)
        else:
            result = await runners[kind](brief, state.get("research"), memory,
                                         feedback=feedback,
                                         prior_draft=prior_draft)
        drafts = {**drafts, kind: result}
        return {**state, "drafts": drafts, "brief": brief,
                "answer": result["draft"]}

    async def series(state: BlitzState) -> BlitzState:
        """Content series generation for campaigns (via workflows)."""
        from contentblitz.workflows import generate_series

        brief = state.get("brief") or {}
        parts = await generate_series(brief, n=3, memory=memory)
        drafts = {**(state.get("drafts") or {})}
        for i, part in enumerate(parts):
            drafts[f"series_part_{i + 1}"] = part
        body = "\n\n---\n\n".join(
            f"## Part {i + 1}: {p['title']}\n\n{p['draft']}"
            for i, p in enumerate(parts))
        return {**state, "drafts": drafts, "answer": body}

    async def chat(state: BlitzState) -> BlitzState:
        answer = await agents._ask_llm(
            "You are ContentBlitz, an AI content-marketing assistant. Answer "
            "briefly in markdown. You can: research topics, write SEO blogs, "
            "write LinkedIn posts, generate images, and build content "
            "strategies. If the user seems to want content, say what you need "
            "(topic, audience) to start.",
            (memory.history_text() if memory else "")
            + f"\n\nUser: {state.get('user_input', '')}",
        )
        if not answer:
            answer = (
                "I'm ContentBlitz, your AI content-marketing assistant. I can:\n\n"
                "- 🔍 Research any topic (with citations)\n"
                "- ✍️ Write SEO-optimized blog articles\n"
                "- 💼 Write LinkedIn posts\n"
                "- 🎨 Generate marketing visuals\n"
                "- 🗺️ Build content strategies\n\n"
                "Give me a topic and audience to get started — "
                "e.g. *'Write a blog post about email marketing for SaaS founders'*."
                + agents.OFFLINE_TAG
            )
        return {**state, "answer": answer}

    # -- wiring ----------------------------------------------------------
    builder.add_node("query_handler", query_handler)
    builder.add_node("deep_research", deep_research)
    builder.add_node("seo_blog", seo_blog)
    builder.add_node("linkedin", linkedin)
    builder.add_node("image", image)
    builder.add_node("strategist", strategist)
    builder.add_node("combo", combo)
    builder.add_node("refine", refine)
    builder.add_node("series", series)
    builder.add_node("chat", chat)

    builder.set_entry_point("query_handler")
    builder.add_conditional_edges("query_handler", _route)
    # Research-first: after research, return to the Query Handler for
    # post-research routing.
    builder.add_edge("deep_research", "query_handler")
    for node in ("seo_blog", "linkedin", "image", "strategist",
                 "combo", "refine", "series", "chat"):
        builder.add_edge(node, END)

    return builder.compile()


def _image_answer(result: dict) -> str:
    if result.get("status") == "generated":
        return (f"🎨 **Image generated** (model: {result.get('model')})\n\n"
                f"![generated visual]({result['image_url']})\n\n"
                f"*Prompt used:* {result.get('prompt', '')[:300]}")
    return (f"🖼️ **Image placeholder** — {result.get('note', '')}\n\n"
            f"*Planned prompt:* {result.get('prompt', '')[:300]}")
