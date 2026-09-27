"""Live integration probes — skipped unless API keys are present.

Run with real keys in the environment to verify the live paths:

    OPENAI_API_KEY=... TAVILY_API_KEY=... SERPAPI_API_KEY=... \\
        pytest tests/test_live_integrations.py -p no:cacheprovider

These tests are intentionally cheap: they probe capabilities without
burning meaningful credit (no image is generated — see
check_image_support; research is a single small query).
"""
import asyncio

import pytest

from contentblitz import tools
from contentblitz.config import settings

needs_openai = pytest.mark.skipif(
    not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
needs_research_key = pytest.mark.skipif(
    not settings.research_available, reason="no research key set")


@needs_research_key
def test_live_web_research_returns_real_provider():
    pack = asyncio.run(tools.web_research("B2B content marketing benchmarks",
                                          max_results=3))
    assert pack["provider"] in ("tavily", "serpapi")
    assert pack["results"]
    assert all(r["url"].startswith("http") for r in pack["results"])


@needs_openai
def test_live_image_support_probe():
    support = tools.check_image_support()
    assert support["key_present"] is True
    # Report honestly; do not fail if the course key lacks DALL-E.
    print(f"\nDALL-E support: 3={support['dall-e-3']} 2={support['dall-e-2']}")


@needs_openai
def test_live_llm_router_decision():
    from contentblitz.agents import route_query

    d = asyncio.run(route_query("write an SEO blog about pricing pages"))
    assert d["intent"] in ("blog", "combo", "research")
    assert d["confidence"] > 0
