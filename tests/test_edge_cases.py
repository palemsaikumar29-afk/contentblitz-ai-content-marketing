"""Edge-case tests for remaining behavior branches."""
import asyncio

from contentblitz import workflows
from contentblitz.agents import route_query_sync
from contentblitz.brand import check_voice_consistency
from contentblitz.quality import (
    flesch_reading_ease,
    make_meta_description,
    validate_content,
)


def test_research_plus_two_formats_is_combo():
    d = route_query_sync("research trends and write a blog and LinkedIn post")
    assert d["intent"] == "combo"
    assert d["needs_research"] is True
    assert set(d["formats"]) == {"blog", "linkedin"}


def test_friendly_voice_too_many_emojis_flagged():
    check = check_voice_consistency("Hi! 🎉🚀🔥💯", "friendly")
    assert check["score"] < 100
    assert any("emojis" in f for f in check["flags"])


def test_professional_voice_hype_words_flagged():
    check = check_voice_consistency(
        "Our revolutionary game-changing platform delivers insane ROI.",
        "professional")
    assert check["score"] < 100
    assert any("hype" in f for f in check["flags"])


def test_bold_voice_long_sentences_flagged():
    long_text = ("We believe that every single marketing team deserves a "
                 "comprehensive platform that unifies all of their workflows "
                 "into one coherent system that scales with their ambitions "
                 "and delivers measurable outcomes across every channel.")
    check = check_voice_consistency(long_text, "bold")
    assert any("too long" in f for f in check["flags"])


def test_flesch_empty_text_returns_zero():
    assert flesch_reading_ease("") == 0.0
    assert flesch_reading_ease("...") == 0.0


def test_meta_description_truncates_long_text():
    long_text = "word " * 200
    meta = make_meta_description(long_text, ["word"])
    assert len(meta) <= 160
    assert meta.endswith("...")


def test_validate_fixes_dangling_colon():
    result = validate_content("# Title\n\nSome intro text here:", "blog")
    assert not result["enhanced"].rstrip().endswith(":")
    assert any("colon" in i for i in result["issues"])


def test_validate_flags_voice_drift():
    result = validate_content(
        "Our revolutionary game-changing insane platform!!! 🚀🚀", "blog",
        "professional")
    assert any("voice drift" in i for i in result["issues"])


def test_generate_series_uses_provided_arc(monkeypatch):
    """The strategist-arc branch: stub _ask_llm to return a fixed arc."""
    import contentblitz.agents as agents_mod

    async def fake_ask(system, user):
        return "1. Hook\n2. Proof\n3. Offer"

    monkeypatch.setattr(agents_mod, "_ask_llm", fake_ask)
    brief = {"topic": "launch", "audience": "founders",
             "brand_voice": "professional"}
    parts = asyncio.run(workflows.generate_series(brief, n=3))
    assert [p["title"] for p in parts] == ["Hook", "Proof", "Offer"]
