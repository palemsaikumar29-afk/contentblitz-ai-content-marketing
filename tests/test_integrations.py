"""Integration-layer tests — HTTP adapters with mocked transports.

No real network calls: requests and the OpenAI client are stubbed, so the
Tavily/SerpApi/DALL-E parsing and fallback behavior is verified hermetically.
"""
import asyncio
from types import SimpleNamespace

import pytest

from contentblitz import tools
from contentblitz.tools import (
    _openai_image_sync,
    _serpapi_search_sync,
    _tavily_search_sync,
    check_image_support,
    generate_image,
    web_research,
)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# --- Tavily adapter -------------------------------------------------------
def test_tavily_parses_results(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        assert "tavily" in url
        return _Resp({"results": [
            {"title": "T1", "url": "https://t1.co", "content": "snippet one"},
            {"title": "T2", "url": "https://t2.co", "content": "snippet two"},
        ]})

    monkeypatch.setattr(tools.requests, "post", fake_post)
    out = _tavily_search_sync("q", 2)
    assert out[0] == {"title": "T1", "url": "https://t1.co",
                      "snippet": "snippet one", "source": "tavily"}


def test_tavily_exception_returns_none(monkeypatch):
    def boom(url, json=None, timeout=None):
        raise ConnectionError("down")

    monkeypatch.setattr(tools.requests, "post", boom)
    assert _tavily_search_sync("q", 2) is None


def test_tavily_empty_results_returns_none(monkeypatch):
    monkeypatch.setattr(tools.requests, "post",
                        lambda *a, **k: _Resp({"results": []}))
    assert _tavily_search_sync("q", 2) is None


# --- SerpApi adapter ------------------------------------------------------
def test_serpapi_parses_results(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert "serpapi" in url
        return _Resp({"organic_results": [
            {"title": "S1", "link": "https://s1.co", "snippet": "s one"},
        ]})

    monkeypatch.setattr(tools.requests, "get", fake_get)
    out = _serpapi_search_sync("q", 3)
    assert out == [{"title": "S1", "url": "https://s1.co",
                    "snippet": "s one", "source": "serpapi"}]


def test_serpapi_exception_returns_none(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise TimeoutError("down")

    monkeypatch.setattr(tools.requests, "get", boom)
    assert _serpapi_search_sync("q", 3) is None


# --- DALL-E adapter -------------------------------------------------------
class _FakeImages:
    def __init__(self, url):
        self._url = url

    def generate(self, model=None, prompt=None, size=None, n=None):
        data = [SimpleNamespace(url=self._url)] if self._url else []
        return SimpleNamespace(data=data)


class _FakeClient:
    def __init__(self, image_url=None, model_ids=()):
        self._image_url = image_url
        self._model_ids = model_ids

    @property
    def images(self):
        return _FakeImages(self._image_url)

    @property
    def models(self):
        outer = self

        class _Models:
            def list(self_inner):
                return [SimpleNamespace(id=m) for m in outer._model_ids]

        return _Models()


def _patch_openai(monkeypatch, **kwargs):
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda api_key=None: _FakeClient(**kwargs))


def test_openai_image_sync_success(monkeypatch):
    _patch_openai(monkeypatch, image_url="https://img.co/1.png")
    out = _openai_image_sync("a robot", "dall-e-3")
    assert out["status"] == "generated"
    assert out["image_url"] == "https://img.co/1.png"
    assert out["model"] == "dall-e-3"


def test_openai_image_sync_no_url_returns_none(monkeypatch):
    _patch_openai(monkeypatch, image_url=None)
    assert _openai_image_sync("a robot", "dall-e-3") is None


def test_openai_image_sync_exception_returns_none(monkeypatch):
    import openai

    def boom(api_key=None):
        raise RuntimeError("auth failed")

    monkeypatch.setattr(openai, "OpenAI", boom)
    assert _openai_image_sync("a robot", "dall-e-3") is None


def test_generate_image_primary_success(monkeypatch):
    import contentblitz.config as cfg

    _patch_openai(monkeypatch, image_url="https://img.co/3.png")
    # The free Pollinations leg is down so the DALL-E path is exercised.
    monkeypatch.setattr(tools, "_pollinations_image_sync", lambda p: None)
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    out = asyncio.run(generate_image("a robot"))
    assert out["status"] == "generated"
    assert out["model"] == "dall-e-3"


def test_generate_image_pollinations_first_success(monkeypatch):
    """The free keyless provider is tried before any DALL-E call."""
    import contentblitz.config as cfg

    class _ImgResp:
        status_code = 200
        headers = {"content-type": "image/jpeg"}

    monkeypatch.setattr(tools.requests, "get", lambda url, timeout=None: _ImgResp())
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    out = asyncio.run(generate_image("a robot"))
    assert out["status"] == "generated"
    assert out["model"] == "pollinations/flux"


def test_check_image_support_lists_models(monkeypatch):
    import contentblitz.config as cfg

    _patch_openai(monkeypatch, model_ids=("dall-e-3", "gpt-4o"))
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    support = check_image_support()
    assert support == {"dall-e-3": True, "dall-e-2": False,
                       "key_present": True, "pollinations": True}


def test_check_image_support_probe_error(monkeypatch):
    import contentblitz.config as cfg
    import openai

    def boom(api_key=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    support = check_image_support()
    assert support["key_present"] is True
    assert support["probe_error"] is True


def test_web_research_prefers_tavily_over_serpapi_live(monkeypatch):
    """End-to-end fallback order with stubbed HTTP (keys present)."""
    import contentblitz.config as cfg

    def fake_post(url, json=None, timeout=None):
        return _Resp({"results": [
            {"title": "T", "url": "https://t.co", "content": "s"}]})

    def fake_get(url, params=None, timeout=None):
        raise AssertionError("SerpApi must not be called when Tavily works")

    monkeypatch.setattr(cfg.settings, "TAVILY_API_KEY", "tv")
    monkeypatch.setattr(cfg.settings, "SERPAPI_API_KEY", "sp")
    monkeypatch.setattr(tools.requests, "post", fake_post)
    monkeypatch.setattr(tools.requests, "get", fake_get)
    pack = asyncio.run(web_research("q", max_results=2))
    assert pack["provider"] == "tavily"
    assert pack["results"][0]["source"] == "tavily"


# --- LLM plumbing ---------------------------------------------------------
def test_ask_llm_uses_chat_openai(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg
    import langchain_openai

    class FakeLLM:
        def __init__(self, model=None, temperature=None):
            pass

        def invoke(self, messages):
            assert messages[0]["role"] == "system"
            return SimpleNamespace(content="llm says hi")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    out = asyncio.run(agents_mod._ask_llm("sys", "user"))
    assert out == "llm says hi"


def test_ask_llm_invoke_error_returns_none(monkeypatch):
    import contentblitz.agents as agents_mod
    import contentblitz.config as cfg
    import langchain_openai

    class BadLLM:
        def __init__(self, model=None, temperature=None):
            pass

        def invoke(self, messages):
            raise RuntimeError("model overloaded")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", BadLLM)
    monkeypatch.setattr(cfg.settings, "OPENAI_API_KEY", "test-key")
    out = asyncio.run(agents_mod._ask_llm("sys", "user"))
    assert out is None


def test_ask_llm_offline_returns_none():
    import contentblitz.agents as agents_mod

    out = asyncio.run(agents_mod._ask_llm("sys", "user"))
    assert out is None  # no key -> None, never raises
