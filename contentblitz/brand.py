"""Brand voice profiles + consistency enforcement.

A brand voice keeps every output — blog, LinkedIn, image prompt — sounding
like the same company. Profiles define tone levers; :func:`voice_guidance`
renders them into writer prompts, and :func:`check_voice_consistency` scores
a draft against the active voice so the quality pipeline can flag drift.
"""
from __future__ import annotations

VOICES: dict[str, dict] = {
    "professional": {
        "label": "Professional",
        "tone": "confident, clear, measured; no slang, no hype",
        "sentence_style": "medium-length sentences, active voice",
        "vocabulary": "precise business language; avoid buzzwords",
        "emoji": "none",
        "cta_style": "direct and polite (e.g. 'Learn more', 'Talk to our team')",
    },
    "friendly": {
        "label": "Friendly",
        "tone": "warm, encouraging, conversational; like a helpful colleague",
        "sentence_style": "varied lengths, contractions welcome",
        "vocabulary": "plain language, light humor allowed",
        "emoji": "sparingly (max 2-3 per post)",
        "cta_style": "inviting (e.g. 'Give it a try', 'We'd love to hear from you')",
    },
    "bold": {
        "label": "Bold",
        "tone": "provocative, opinionated, energetic; takes a stance",
        "sentence_style": "short punchy sentences, fragments for emphasis",
        "vocabulary": "strong verbs, vivid metaphors",
        "emoji": "ok for social, never in long-form",
        "cta_style": "challenging (e.g. 'Stop doing X. Start doing Y.')",
    },
    "technical": {
        "label": "Technical",
        "tone": "precise, evidence-driven, neutral",
        "sentence_style": "structured, numbered points, defined terms",
        "vocabulary": "domain terminology with brief definitions on first use",
        "emoji": "none",
        "cta_style": "resource-oriented (e.g. 'Read the docs', 'See the benchmark')",
    },
}

DEFAULT_VOICE = "professional"


def get_voice(name: str | None) -> dict:
    """Return the voice profile, falling back to the default."""
    key = (name or DEFAULT_VOICE).lower()
    voice = VOICES.get(key, VOICES[DEFAULT_VOICE])
    return {"name": key if key in VOICES else DEFAULT_VOICE, **voice}


def voice_guidance(name: str | None) -> str:
    """Render a voice profile as writer instructions."""
    v = get_voice(name)
    return (
        f"Brand voice: {v['label']}. Tone: {v['tone']}. "
        f"Sentence style: {v['sentence_style']}. Vocabulary: {v['vocabulary']}. "
        f"Emoji policy: {v['emoji']}. CTA style: {v['cta_style']}."
    )


def check_voice_consistency(text: str, voice_name: str | None) -> dict:
    """Heuristic voice-drift check. Returns score 0-100 + flags.

    Rule-based (no LLM needed): each voice has cheap textual markers.
    The quality pipeline uses this to flag drafts that drift off-brand.
    """
    v = get_voice(voice_name)
    name = v["name"]
    lowered = text.lower()
    flags: list[str] = []
    score = 100

    emoji_count = sum(1 for ch in text if ord(ch) > 0x1F300)
    if name in ("professional", "technical") and emoji_count > 0:
        flags.append(f"emoji used in '{name}' voice")
        score -= 20
    if name == "friendly" and emoji_count > 3:
        flags.append("too many emojis for 'friendly' voice")
        score -= 10

    hype_words = ["revolutionary", "game-changing", "mind-blowing", "insane",
                  "crazy", "unbelievable"]
    hype_hits = [w for w in hype_words if w in lowered]
    if name in ("professional", "technical") and hype_hits:
        flags.append(f"hype words {hype_hits} clash with '{name}' voice")
        score -= 15 * len(hype_hits)

    if name == "bold":
        sentences = [s for s in text.replace("!", ".").split(".") if s.strip()]
        avg_len = (sum(len(s.split()) for s in sentences) / len(sentences)
                   if sentences else 0)
        if avg_len > 28:
            flags.append("sentences too long for 'bold' voice")
            score -= 10

    return {"voice": name, "score": max(0, score), "flags": flags}
