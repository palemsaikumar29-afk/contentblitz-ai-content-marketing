"""Workflow tests — research-first, series, refinement (hermetic)."""
import asyncio

import pytest

from contentblitz.memory import ConversationMemory
from contentblitz.workflows import (
    generate_series,
    refine_content,
    run_research_first,
)

BRIEF = {"topic": "email marketing", "audience": "SaaS founders",
         "brand_voice": "professional"}


def test_run_research_first_all_formats():
    mem = ConversationMemory()
    pack = asyncio.run(run_research_first(BRIEF, mem))
    assert pack["research"]["citation_count"] >= 3
    assert set(pack["outputs"]) == {"blog", "linkedin", "strategy"}
    assert pack["outputs"]["blog"]["seo"]["keywords"]
    assert mem.last_output("blog")
    assert mem.last_research


def test_run_research_first_subset():
    pack = asyncio.run(run_research_first(BRIEF, formats=("linkedin",)))
    assert set(pack["outputs"]) == {"linkedin"}


def test_generate_series_parts_and_scores():
    mem = ConversationMemory()
    parts = asyncio.run(generate_series(BRIEF, n=3, memory=mem))
    assert len(parts) == 3
    assert [p["part"] for p in parts] == [1, 2, 3]
    assert all(p["title"] for p in parts)
    assert all(p["series_score"] > 0 for p in parts)
    assert mem.last_output("series_part_1")


def test_refine_content_improves_or_keeps_best():
    mem = ConversationMemory()
    original = "Marketing is very really just quite good. " * 10
    out = asyncio.run(refine_content("linkedin", original,
                                     "make it punchier with a CTA",
                                     brief=BRIEF, memory=mem,
                                     max_rounds=2))
    assert out["kind"] == "linkedin"
    assert out["rounds_run"] >= 1
    assert out["draft"]


def test_refine_content_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown content kind"):
        asyncio.run(refine_content("podcast", "draft", "feedback"))
