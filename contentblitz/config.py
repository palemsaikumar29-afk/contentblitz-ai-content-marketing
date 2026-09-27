"""Central configuration. All secrets come from environment variables only.

Required for full functionality:
    OPENAI_API_KEY   Chat/completion model + DALL-E image generation
Optional (tools degrade gracefully when missing):
    TAVILY_API_KEY   Web research (preferred)
    SERPAPI_API_KEY  Web research fallback

Without keys the pipeline runs in offline/demo mode: deterministic
rule-based writers, mock research with labelled citations, and a
clearly-labelled image placeholder (never presented as real).
"""
from __future__ import annotations

import os


def _sanitize_proxy_env() -> None:
    """Drop IPv6 literals from no_proxy that crash httpx-based clients.

    Some runtimes export IPv6 addresses (e.g. ``::1``, ``[::1]``) in
    no_proxy. httpx turns each entry into a mount pattern like
    ``all://*[::1]`` which its own URLPattern parser then rejects with
    ``InvalidURL: Invalid port``. IPv6 loopback entries are irrelevant for
    this app's outbound API calls, so they are removed; everything else is
    left untouched.
    """
    for var in ("no_proxy", "NO_PROXY"):
        raw = os.environ.get(var)
        if not raw:
            continue
        kept = [
            entry.strip()
            for entry in raw.split(",")
            if ":" not in entry  # drop IPv6 literals, keep hostnames/IPv4
        ]
        os.environ[var] = ",".join(e for e in kept if e)


_sanitize_proxy_env()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")

    LLM_MODEL: str = os.getenv("CONTENTBLITZ_LLM_MODEL", "gpt-4o-mini")
    # DALL-E fallback chain: 3 -> 2 -> labelled placeholder.
    IMAGE_MODEL_PRIMARY: str = os.getenv("CONTENTBLITZ_IMAGE_MODEL", "dall-e-3")
    IMAGE_MODEL_FALLBACK: str = "dall-e-2"
    RESEARCH_MAX_RESULTS: int = int(os.getenv("CONTENTBLITZ_RESEARCH_MAX", "8"))
    MAX_REFINEMENT_ROUNDS: int = int(os.getenv("CONTENTBLITZ_MAX_REFINES", "3"))

    @property
    def llm_available(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def research_available(self) -> bool:
        return bool(self.TAVILY_API_KEY or self.SERPAPI_API_KEY)


settings = Settings()
