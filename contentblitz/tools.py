"""External tools with graceful degradation.

Design rule (same as Finnie): a missing key or a failed request must NEVER
crash the pipeline. Every tool falls back and labels its result with
``source`` so the UI stays transparent:

- Web research: Tavily (preferred) -> SerpApi (fallback) -> mock (labelled)
- Image gen:   Pollinations.ai (free, keyless, default; 3 tries w/ backoff)
               -> DALL-E 3 -> DALL-E 2 -> clearly-labelled placeholder

A placeholder image is NEVER presented as a real generated image.
"""
from __future__ import annotations

import asyncio
import random
import time
from urllib.parse import quote, urlencode

import requests

from contentblitz.config import settings


# ---------------------------------------------------------------------------
# Web research
# ---------------------------------------------------------------------------
def _tavily_search_sync(query: str, max_results: int) -> list[dict] | None:
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=20,
        )
        data = resp.json()
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:500],
                "source": "tavily",
            }
            for r in data.get("results", [])
        ]
        return results or None
    except Exception:
        return None


def _serpapi_search_sync(query: str, max_results: int) -> list[dict] | None:
    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "api_key": settings.SERPAPI_API_KEY,
                "q": query,
                "num": max_results,
            },
            timeout=20,
        )
        data = resp.json()
        organic = data.get("organic_results", [])
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", "")[:500],
                "source": "serpapi",
            }
            for r in organic[:max_results]
        ]
        return results or None
    except Exception:
        return None


def _mock_research(query: str, max_results: int) -> list[dict]:
    """Deterministic labelled mock results for offline/demo mode."""
    topics = [
        (
            f"{query.title()} — industry overview",
            "https://example.com/research/overview",
            f"Background context and definitions for '{query}', useful for "
            "framing the content brief and establishing key terms.",
        ),
        (
            f"Latest trends in {query}",
            "https://example.com/research/trends",
            f"Recent developments and trend analysis around '{query}' with "
            "statistics commonly cited in marketing content.",
        ),
        (
            f"{query.title()}: best practices guide",
            "https://example.com/research/best-practices",
            f"Practitioner best practices for '{query}', including common "
            "pitfalls and recommended approaches.",
        ),
        (
            f"Case studies: {query} in action",
            "https://example.com/research/case-studies",
            f"Real-world examples of '{query}' applied successfully, with "
            "outcomes and lessons learned.",
        ),
    ]
    return [
        {"title": t, "url": u, "snippet": s, "source": "mock"}
        for t, u, s in topics[:max_results]
    ]


async def web_research(query: str, max_results: int | None = None) -> dict:
    """Research a query. Returns {results, provider, query}.

    Provider is one of "tavily" | "serpapi" | "mock".
    """
    n = max_results or settings.RESEARCH_MAX_RESULTS
    results: list[dict] | None = None
    provider = "mock"
    if settings.TAVILY_API_KEY:
        results = await asyncio.to_thread(_tavily_search_sync, query, n)
        if results:
            provider = "tavily"
    if not results and settings.SERPAPI_API_KEY:
        results = await asyncio.to_thread(_serpapi_search_sync, query, n)
        if results:
            provider = "serpapi"
    if not results:
        results = _mock_research(query, n)
    return {"results": results, "provider": provider, "query": query}


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def optimize_image_prompt(brief: dict) -> str:
    """Build a detailed, style-directed image prompt from a content brief."""
    topic = brief.get("topic", "abstract business concept")
    audience = brief.get("audience", "professionals")
    style = brief.get("image_style", "modern flat illustration")
    mood = brief.get("mood", "optimistic and professional")
    brand = brief.get("brand", "")
    brand_bit = f" for the brand '{brand}'" if brand else ""
    return (
        f"{style} depicting '{topic}'{brand_bit}, aimed at {audience}. "
        f"Mood: {mood}. Clean composition, strong focal point, generous "
        f"negative space for text overlay, no readable text or words in the "
        f"image, high detail, professional marketing quality, 16:9."
    )


# Free, keyless default provider — https://image.pollinations.ai/prompt/{prompt}.
# A GET with the prompt URL-encoded in the path returns the generated image
# bytes directly; no signup, no API key, no credit spend. ``private=true``
# keeps generations out of the public feed. The URL carries its own seed,
# so the same URL always returns the same image (cheap re-fetch by the UI).
POLLINATIONS_IMAGE_BASE = "https://image.pollinations.ai/prompt"


def _pollinations_url(prompt: str, seed: int | None = None,
                      width: int = 1024, height: int = 1024) -> str:
    """Build the Pollinations.ai generation URL for a prompt."""
    seed = seed if seed is not None else random.randint(0, 999_999)
    params = urlencode({
        "model": "flux",
        "width": width,
        "height": height,
        "seed": seed,
        "private": "true",
    })
    return f"{POLLINATIONS_IMAGE_BASE}/{quote(prompt, safe='')}?{params}"


def _sleep_between_attempts(seconds: float) -> None:
    """Sleep hook kept separate so tests can observe/patch the backoff."""

    time.sleep(seconds)


# Pollinations.ai is free and keyless, but its free tier is occasionally
# flaky (transient HTTP 500s). Retry a few times with exponential backoff
# before falling through to the DALL-E chain or the labelled placeholder.
POLLINATIONS_MAX_ATTEMPTS = 3
POLLINATIONS_RETRY_BASE_DELAY = 1.0  # seconds; doubled after each failure


def _pollinations_image_sync(prompt: str) -> dict | None:
    """Blocking Pollinations.ai call — always run in a thread.

    Keyless: needs no secrets. Returns None on any failure (timeout, HTTP
    error, or a non-image response) so the caller can fall through to the
    DALL-E chain or the labelled placeholder.
    """
    try:
        url = _pollinations_url(prompt)
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            return None
        if not resp.headers.get("content-type", "").startswith("image/"):
            return None
        return {
            "status": "generated",
            "image_url": url,
            "model": "pollinations/flux",
            "prompt": prompt,
        }
    except Exception:
        return None


def _pollinations_image_sync_with_retry(prompt: str) -> dict | None:
    """Pollinations.ai call with retries and exponential backoff.

    Keyless: needs no secrets. Tries up to ``POLLINATIONS_MAX_ATTEMPTS``
    times, sleeping ``POLLINATIONS_RETRY_BASE_DELAY`` seconds after the
    first failure and doubling the wait after each subsequent failure
    (1s, 2s by default). Returns the generated-image dict on the first
    success, or None when every attempt fails so the caller can fall
    through to the DALL-E chain or the labelled placeholder.
    """
    for attempt in range(POLLINATIONS_MAX_ATTEMPTS):
        result = _pollinations_image_sync(prompt)
        if result:
            return result
        if attempt < POLLINATIONS_MAX_ATTEMPTS - 1:
            _sleep_between_attempts(POLLINATIONS_RETRY_BASE_DELAY * 2**attempt)
    return None


def _openai_image_sync(prompt: str, model: str) -> dict | None:
    """Blocking DALL-E call — always run in a thread. Returns None on any failure."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        size = "1024x1024" if model == "dall-e-2" else "1792x1024"
        resp = client.images.generate(model=model, prompt=prompt, size=size, n=1)
        url = resp.data[0].url if resp.data else None
        if not url:
            return None
        return {
            "status": "generated",
            "image_url": url,
            "model": model,
            "prompt": prompt,
        }
    except Exception:
        return None


def check_image_support() -> dict:
    """Probe which image providers are available.

    Pollinations.ai is free and keyless, so it is always reported as
    available (a generation can still fail at request time, in which case
    the chain falls through to DALL-E / the placeholder). DALL-E models
    are probed against the OpenAI key via a free /v1/models call.
    """
    support = {"pollinations": True}
    if not settings.OPENAI_API_KEY:
        support.update({"dall-e-3": False, "dall-e-2": False,
                        "key_present": False})
        return support
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        ids = {m.id for m in client.models.list()}
        support.update({
            "dall-e-3": "dall-e-3" in ids,
            "dall-e-2": "dall-e-2" in ids,
            "key_present": True,
        })
        return support
    except Exception:
        support.update({"dall-e-3": False, "dall-e-2": False,
                        "key_present": True, "probe_error": True})
        return support


async def generate_image(prompt: str) -> dict:
    """Generate an image with the Pollinations -> DALL-E 3 -> DALL-E 2 ->
    placeholder chain.

    Pollinations.ai is the free, keyless default. DALL-E stays as the paid
    fallback when an OpenAI key is present. Always returns a dict with
    ``status`` in {"generated", "placeholder"}. A placeholder is explicitly
    labelled and never presented as real.
    """
    result = await asyncio.to_thread(_pollinations_image_sync_with_retry, prompt)
    if result:
        return result
    if settings.llm_available:
        result = await asyncio.to_thread(
            _openai_image_sync, prompt, settings.IMAGE_MODEL_PRIMARY
        )
        if result:
            return result
        if settings.IMAGE_MODEL_FALLBACK != settings.IMAGE_MODEL_PRIMARY:
            result = await asyncio.to_thread(
                _openai_image_sync, prompt, settings.IMAGE_MODEL_FALLBACK
            )
            if result:
                return result
    return {
        "status": "placeholder",
        "image_url": None,
        "model": None,
        "prompt": prompt,
        "note": (
            "PLACEHOLDER — image generation is unavailable "
            "(Pollinations.ai and DALL-E calls both failed). "
            "This is not a real generated image."
        ),
    }
