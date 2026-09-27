"""Content quality: scoring, SEO engine, validation + enhancement.

The pipeline every draft passes through before it reaches the user:

    validate_content -> score_content -> seo_optimize -> (voice check)

All heuristics are deterministic so tests are stable with or without an
LLM key.
"""
from __future__ import annotations

import re
from collections import Counter

from contentblitz.brand import check_voice_consistency


# ---------------------------------------------------------------------------
# Text statistics helpers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "by", "as", "at", "from",
    "that", "this", "it", "its", "you", "your", "we", "our", "they", "their",
    "will", "can", "has", "have", "had", "not", "but", "if", "so", "do",
    "does", "into", "than", "then", "them", "he", "she", "his", "her",
}


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = sum(1 for i, ch in enumerate(word)
                if ch in vowels and (i == 0 or word[i - 1] not in vowels))
    return max(1, count - (1 if word.endswith("e") and count > 1 else 0))


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease (higher = easier). Deterministic."""
    words = _words(text)
    sents = _sentences(text)
    if not words or not sents:
        return 0.0
    wps = len(words) / len(sents)
    spw = sum(_syllables(w) for w in words) / len(words)
    return round(206.835 - 1.015 * wps - 84.6 * spw, 1)


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------
TARGET_LENGTHS = {
    "blog": (800, 2500),
    "linkedin": (80, 350),
    "strategy": (300, 1500),
}


def score_content(text: str, content_type: str = "blog") -> dict:
    """Score a draft 0-100 with a per-dimension breakdown + suggestions."""
    text = text or ""
    words = _words(text)
    n_words = len(words)
    breakdown: dict[str, float] = {}
    suggestions: list[str] = []

    # 1. Length appropriateness (25 pts)
    lo, hi = TARGET_LENGTHS.get(content_type, (200, 2000))
    if lo <= n_words <= hi:
        breakdown["length"] = 25.0
    elif n_words < lo:
        breakdown["length"] = round(25 * n_words / lo, 1)
        suggestions.append(
            f"Too short ({n_words} words) for {content_type}; aim for {lo}-{hi}.")
    else:
        breakdown["length"] = round(max(5.0, 25 * hi / n_words), 1)
        suggestions.append(
            f"Long ({n_words} words) for {content_type}; consider tightening.")

    # 2. Structure (25 pts): headers for blog, hook+CTA for LinkedIn.
    headers = re.findall(r"^#{1,3}\s+.+", text, re.M)
    if content_type == "blog":
        breakdown["structure"] = 25.0 if len(headers) >= 3 else len(headers) / 3 * 25
        if len(headers) < 3:
            suggestions.append("Add more section headers (H2/H3) for scannability.")
    elif content_type == "linkedin":
        has_hook = bool(_sentences(text)[:1])
        has_cta = bool(re.search(
            r"(comment|share|follow|link in|dm|message me|learn more|try|join)",
            text.lower()))
        breakdown["structure"] = round(
            12.5 * has_hook + 12.5 * has_cta, 1)
        if not has_cta:
            suggestions.append("LinkedIn posts perform better with a clear CTA.")
    else:
        breakdown["structure"] = 25.0 if headers else 12.5
        if not headers:
            suggestions.append("Add headers to organize the strategy doc.")

    # 3. Readability (25 pts): Flesch 50-70 is the marketing sweet spot.
    fre = flesch_reading_ease(text)
    if 50 <= fre <= 75:
        breakdown["readability"] = 25.0
    else:
        dist = min(abs(fre - 50), abs(fre - 75))
        breakdown["readability"] = round(max(5.0, 25 - dist / 2), 1)
        suggestions.append(
            f"Reading ease {fre} is outside the 50-75 sweet spot; "
            "simplify long sentences.")

    # 4. Engagement signals (25 pts): questions, specifics, no filler.
    signals = 0
    if "?" in text:
        signals += 1
    if re.search(r"\d+[%+]|\d+x", text):
        signals += 1  # concrete numbers
    filler = len(re.findall(
        r"\b(very|really|just|quite|basically|essentially|in order to)\b",
        text.lower()))
    breakdown["engagement"] = round(
        25 * (signals / 2) - min(10, filler), 1)
    breakdown["engagement"] = max(0.0, min(25.0, breakdown["engagement"]))
    if filler > 5:
        suggestions.append(f"Cut filler words ({filler} found: very/really/just...).")

    total = round(sum(breakdown.values()), 1)
    return {
        "score": total,
        "grade": "A" if total >= 85 else "B" if total >= 70 else
                 "C" if total >= 55 else "D",
        "breakdown": breakdown,
        "suggestions": suggestions,
        "word_count": n_words,
        "reading_ease": fre,
    }


# ---------------------------------------------------------------------------
# SEO optimization engine
# ---------------------------------------------------------------------------
def extract_keywords(research: list[dict], topic: str, top_n: int = 8) -> list[str]:
    """Candidate keyword research from research snippets + topic.

    Frequency-based over non-stopwords in titles/snippets, topic terms
    boosted. Deterministic; an LLM pass can refine when a key is present.
    """
    corpus = " ".join(
        f"{r.get('title', '')} {r.get('snippet', '')}" for r in research
    )
    topic_terms = {w for w in _words(topic) if w not in _STOPWORDS}
    counts = Counter(w for w in _words(corpus)
                     if w not in _STOPWORDS and len(w) > 3)
    for w in topic_terms:
        counts[w] += 5
    return [w for w, _ in counts.most_common(top_n)]


def make_meta_description(text: str, keywords: list[str]) -> str:
    """~155-char meta description, keyword-led."""
    first = (_sentences(text)[:1] or [""])[0].strip()
    kw = keywords[0] if keywords else ""
    candidate = f"{kw.capitalize()}: {first}" if kw else first
    if len(candidate) <= 155:
        return candidate
    cut = candidate[:152].rsplit(" ", 1)[0]
    return cut + "..."


def slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "untitled"


def optimize_headers(text: str, keywords: list[str]) -> str:
    """Ensure an H1 exists and H2s carry keywords where missing.

    Never rewrites body copy — only promotes/demotes header lines.
    """
    lines = text.splitlines()
    has_h1 = any(l.startswith("# ") for l in lines)
    out: list[str] = []
    h2_count = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") and keywords:
            h2_count += 1
            low = stripped.lower()
            # If the H2 has no keyword and we still have spares, append one.
            if not any(k in low for k in keywords[:5]) and h2_count <= len(keywords):
                kw = keywords[(h2_count - 1) % len(keywords)]
                line = f"{stripped.rstrip()} — {kw}"
        out.append(line)
        if i == 0 and not has_h1 and stripped and not stripped.startswith("#"):
            # Promote a leading title-like first line to H1.
            out[0] = f"# {stripped}"
            has_h1 = True
    return "\n".join(out)


def seo_optimize(draft: str, brief: dict, research: list[dict]) -> dict:
    """Full SEO pass: keywords, meta description, headers, slug, density."""
    topic = brief.get("topic", "")
    keywords = extract_keywords(research, topic)
    optimized = optimize_headers(draft, keywords)
    words = _words(optimized)
    density = {}
    if words:
        for kw in keywords[:5]:
            hits = sum(1 for w in words if kw in w)
            density[kw] = round(100 * hits / len(words), 2)
    return {
        "optimized": optimized,
        "keywords": keywords,
        "meta_description": make_meta_description(optimized, keywords),
        "slug": slugify(topic),
        "keyword_density_pct": density,
    }


# ---------------------------------------------------------------------------
# Validation + enhancement pipeline
# ---------------------------------------------------------------------------
def validate_content(text: str, content_type: str = "blog",
                     voice: str | None = None) -> dict:
    """Validate a draft and auto-enhance fixable issues.

    Returns {valid, issues, enhanced, score, voice_check}.
    """
    issues: list[str] = []
    enhanced = text or ""

    if not enhanced.strip():
        return {"valid": False, "issues": ["empty draft"],
                "enhanced": enhanced, "score": None, "voice_check": None}

    # Fix 1: collapse 3+ blank lines.
    fixed = re.sub(r"\n{3,}", "\n\n", enhanced)
    if fixed != enhanced:
        issues.append("collapsed excessive blank lines")
        enhanced = fixed

    # Fix 2: LinkedIn length guard.
    words = _words(enhanced)
    if content_type == "linkedin" and len(words) > 400:
        issues.append("linkedin draft exceeds ~400 words; trimmed")
        enhanced = " ".join(enhanced.split()[:380]) + "…"

    # Fix 3: ensure blog ends cleanly (no dangling colon).
    if content_type == "blog" and enhanced.rstrip().endswith(":"):
        issues.append("removed dangling trailing colon")
        enhanced = enhanced.rstrip()[:-1].rstrip() + "."

    score = score_content(enhanced, content_type)
    voice_check = check_voice_consistency(enhanced, voice)
    if voice_check["flags"]:
        issues.append("voice drift: " + "; ".join(voice_check["flags"]))

    return {
        "valid": not issues or score["score"] >= 40,
        "issues": issues,
        "enhanced": enhanced,
        "score": score,
        "voice_check": voice_check,
    }
