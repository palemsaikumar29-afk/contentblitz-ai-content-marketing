"""Quality + brand voice tests (deterministic heuristics)."""
from contentblitz.brand import (
    check_voice_consistency,
    get_voice,
    voice_guidance,
)
from contentblitz.quality import (
    extract_keywords,
    flesch_reading_ease,
    make_meta_description,
    optimize_headers,
    score_content,
    seo_optimize,
    slugify,
    validate_content,
)

GOOD_BLOG = """# Email Marketing in 2026

Email remains the highest-ROI channel for SaaS teams.

## Why it still wins

With a 36% average open rate, email outperforms social reach by 5x.
Teams that segment see 760% more revenue from campaigns.

## Getting started

### Build the list

Offer a lead magnet. Keep forms short.

### Write better subject lines

Questions lift opens by 21%. Test two variants weekly.

## FAQ

**How often should we send?** Weekly is the sweet spot for most teams.

## Conclusion

Start with one welcome sequence this week. Measure, then iterate.
What will you test first?"""

THIN_TEXT = "Marketing is very really just quite good."


def test_flesch_sane_range():
    score = flesch_reading_ease(GOOD_BLOG)
    assert 30 <= score <= 100


def test_score_good_blog_high():
    result = score_content(GOOD_BLOG, "blog")
    assert result["score"] >= 60
    assert result["grade"] in ("A", "B")
    assert result["word_count"] > 50


def test_score_thin_text_low_with_suggestions():
    result = score_content(THIN_TEXT, "blog")
    assert result["score"] < 60
    assert result["suggestions"]  # actionable feedback present


def test_score_linkedin_cta():
    with_cta = ("New playbook just dropped.\n\nWe grew trials 40% with one "
                "change.\n\nWhat worked for you? Comment below. #Growth")
    no_cta = "New playbook just dropped. We grew trials 40% with one change."
    assert (score_content(with_cta, "linkedin")["score"]
            >= score_content(no_cta, "linkedin")["score"])


def test_extract_keywords_boosts_topic_terms():
    research = [
        {"title": "Email benchmarks", "snippet": "email open rates rose"},
        {"title": "Deliverability guide", "snippet": "email deliverability tips"},
    ]
    kws = extract_keywords(research, "email marketing", top_n=5)
    assert "email" in kws
    assert len(kws) == 5


def test_meta_description_length_and_keyword():
    meta = make_meta_description(GOOD_BLOG, ["email", "marketing"])
    assert len(meta) <= 160
    assert "email" in meta.lower()


def test_slugify():
    assert slugify("Email Marketing in 2026!") == "email-marketing-in-2026"


def test_optimize_headers_adds_h1_and_keywords():
    text = "My Title\n\n## Section one\n\nBody.\n\n## Section two\n\nMore."
    out = optimize_headers(text, ["email", "marketing"])
    assert out.startswith("# My Title")
    assert "email" in out.lower() or "marketing" in out.lower()


def test_seo_optimize_full_pass():
    brief = {"topic": "email marketing"}
    research = [{"title": "Email benchmarks",
                 "snippet": "email open rates and email deliverability data"}]
    out = seo_optimize(GOOD_BLOG, brief, research)
    assert out["keywords"]
    assert len(out["meta_description"]) <= 160
    assert out["slug"] == "email-marketing"
    assert out["keyword_density_pct"]


def test_validate_empty_invalid():
    result = validate_content("   ", "blog")
    assert result["valid"] is False
    assert "empty draft" in result["issues"]


def test_validate_collapses_blank_lines():
    result = validate_content("Hello\n\n\n\nworld", "blog")
    assert "\n\n\n" not in result["enhanced"]
    assert any("blank lines" in i for i in result["issues"])


def test_validate_trims_long_linkedin():
    long_post = "word " * 500
    result = validate_content(long_post, "linkedin")
    assert len(result["enhanced"].split()) <= 381


def test_validate_returns_score_and_voice():
    result = validate_content(GOOD_BLOG, "blog", "professional")
    assert result["score"]["score"] > 0
    assert result["voice_check"]["voice"] == "professional"


def test_brand_voice_guidance():
    guidance = voice_guidance("bold")
    assert "Bold" in guidance
    assert "CTA" in guidance or "cta" in guidance.lower()


def test_get_voice_unknown_falls_back():
    assert get_voice("nonexistent")["name"] == "professional"


def test_voice_consistency_flags_emoji_in_professional():
    check = check_voice_consistency("Great results! 🚀🚀", "professional")
    assert check["score"] < 100
    assert check["flags"]


def test_voice_consistency_clean_professional():
    check = check_voice_consistency(
        "Our platform helps teams ship faster with clear reporting.",
        "professional")
    assert check["score"] == 100
    assert not check["flags"]
