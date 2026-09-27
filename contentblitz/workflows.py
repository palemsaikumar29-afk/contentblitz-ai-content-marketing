"""Advanced multi-agent workflows.

- :func:`run_research_first` — deep research -> multi-format content
  (blog + LinkedIn + strategy), the flagship research-first workflow.
- :func:`generate_series` — content series generation for campaigns:
  strategist plans the arc, then each part is drafted (and scored).
- :func:`refine_content` — multi-turn iterative refinement of a draft.
"""
from __future__ import annotations

from contentblitz import agents
from contentblitz.agents import (
    content_strategist,
    deep_research_agent,
    linkedin_writer,
    seo_blog_writer,
)
from contentblitz.quality import score_content


async def run_research_first(brief: dict, memory=None,
                             formats: tuple[str, ...] = ("blog", "linkedin",
                                                          "strategy")) -> dict:
    """Research-first workflow: research once, produce every format."""
    research = await deep_research_agent(brief, memory)
    outputs: dict[str, dict] = {}
    if "blog" in formats:
        outputs["blog"] = await seo_blog_writer(brief, research, memory)
    if "linkedin" in formats:
        outputs["linkedin"] = await linkedin_writer(brief, research, memory)
    if "strategy" in formats:
        outputs["strategy"] = await content_strategist(brief, research, memory)
    return {"research": research, "outputs": outputs}


async def generate_series(brief: dict, n: int = 3, memory=None,
                          research: dict | None = None) -> list[dict]:
    """Content series for campaigns.

    Step 1: the strategist defines the series arc (n part-themes).
    Step 2: each part is drafted as a LinkedIn-style post and scored.
    """
    topic = brief.get("topic", "Untitled")
    arc = await agents._ask_llm(
        "You are a content strategist. Given a topic, propose a numbered list "
        "of exactly the requested number of distinct part-themes for a "
        "social content series. Reply with ONLY the numbered list.",
        f"Topic: {topic}\nAudience: {brief.get('audience', '')}\n"
        f"Number of parts: {n}",
    )
    if arc:
        import re
        themes = [re.sub(r"^\d+[\).\s]+", "", line).strip()
                  for line in arc.splitlines() if line.strip()][:n]
    else:
        themes = []
    while len(themes) < n:  # offline fallback themes
        themes.append(f"{topic} — angle {len(themes) + 1}: lessons learned")
    if research is None:
        research = await deep_research_agent(brief, memory)

    parts: list[dict] = []
    for i, theme in enumerate(themes[:n]):
        part_brief = {**brief, "topic": theme}
        out = await linkedin_writer(part_brief, research, memory)
        out["title"] = theme
        out["part"] = i + 1
        out["series_score"] = score_content(out["draft"], "linkedin")["score"]
        parts.append(out)
        if memory is not None:
            memory.remember_output(f"series_part_{i + 1}", out["draft"])
    return parts


async def refine_content(kind: str, draft: str, feedback: str,
                         brief: dict | None = None,
                         research: dict | None = None,
                         memory=None, max_rounds: int | None = None) -> dict:
    """Iterative refinement: apply feedback, re-validate, re-score.

    Runs up to ``max_rounds`` passes; stops early when the quality score
    stops improving.
    """
    from contentblitz.config import settings

    brief = brief or {}
    rounds = max_rounds or settings.MAX_REFINEMENT_ROUNDS
    writers = {"blog": seo_blog_writer, "linkedin": linkedin_writer,
               "strategy": content_strategist}
    writer = writers.get(kind)
    if writer is None:
        raise ValueError(f"Unknown content kind: {kind!r}")
    history = [{"draft": draft,
                "score": score_content(draft, kind)["score"]}]
    current = draft
    for _ in range(rounds):
        out = await writer(brief, research, memory, feedback=feedback)
        new_score = out["quality"]["score"]
        history.append({"draft": out["draft"], "score": new_score})
        if new_score <= history[-2]["score"]:
            break  # no improvement — keep the previous best
        current = out["draft"]
    best = max(history, key=lambda h: h["score"])
    return {"kind": kind, "draft": best["draft"], "score": best["score"],
            "rounds_run": len(history) - 1, "feedback": feedback}
