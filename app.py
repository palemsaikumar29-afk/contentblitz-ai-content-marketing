"""ContentBlitz — AI Content Marketing Agent. Streamlit UI.

Four tabs:
- 💬 Chat        conversational interface, multi-turn memory
- 📝 Dashboard   brief -> pipeline -> preview AND edit before publishing
- 🔍 Research    deep-research panel with cited results
- 📤 Export      download drafts as Markdown / HTML / PDF

Keys via environment (all optional — the app degrades gracefully):
    OPENAI_API_KEY   LLM writers + DALL-E image fallback
    TAVILY_API_KEY   web research (preferred)
    SERPAPI_API_KEY  web research fallback

Image generation is free via Pollinations.ai and needs no key.
"""
from __future__ import annotations

import asyncio

import streamlit as st

import contentblitz
from contentblitz.agents import (
    content_strategist,
    deep_research_agent,
    image_agent,
    linkedin_writer,
    seo_blog_writer,
)
from contentblitz.exports import export_html, export_pdf
from contentblitz.brand import VOICES
from contentblitz.config import settings
from contentblitz.graph import build_graph
from contentblitz.memory import ConversationMemory
from contentblitz.quality import score_content, validate_content
from contentblitz.workflows import run_research_first

st.set_page_config(page_title="ContentBlitz — AI Content Marketing",
                   page_icon="✍️", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _state():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(st.session_state.memory)
    if "drafts" not in st.session_state:
        st.session_state.drafts = {}   # kind -> {"draft": str, "quality": dict, ...}
    if "research" not in st.session_state:
        st.session_state.research = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    return st.session_state


def _run(coro):
    return asyncio.run(coro)


def _brief_from_sidebar() -> dict:
    return {
        "topic": st.session_state.get("sb_topic", "").strip(),
        "audience": st.session_state.get("sb_audience", "").strip(),
        "brand": st.session_state.get("sb_brand", "").strip(),
        "brand_voice": st.session_state.get("sb_voice", "professional"),
        "image_style": st.session_state.get("sb_style",
                                            "modern flat illustration"),
        "mood": st.session_state.get("sb_mood", "optimistic and professional"),
    }


def _store_draft(kind: str, result: dict):
    st.session_state.drafts[kind] = result


def _quality_badge(quality: dict | None) -> str:
    if not quality:
        return ""
    return f"**Quality: {quality['score']}/100 (grade {quality['grade']})**"


S = _state()
MEM: ConversationMemory = S.memory

st.title("✍️ ContentBlitz — AI Content Marketing")
st.caption(f"Six-agent content pipeline · v{contentblitz.__version__}")

# ---------------------------------------------------------------------------
# Sidebar: status + brief defaults
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Status")
    st.write("🤖 LLM:", "online" if settings.llm_available else "offline mode")
    st.write("🔍 Research:",
             "live" if settings.research_available else "demo data")
    st.write("🎨 Image gen:",
             "Pollinations (free)" + (" + DALL-E" if settings.llm_available
                                      else ""))
    st.caption("Image generation is free via Pollinations.ai — no key needed. "
               "Add API keys via environment variables to unlock live "
               "LLM writing and real research.")
    st.divider()
    st.header("Brief defaults")
    st.text_input("Topic", key="sb_topic",
                  placeholder="e.g. Launching a SaaS analytics feature")
    st.text_input("Audience", key="sb_audience",
                  placeholder="e.g. B2B product managers")
    st.text_input("Brand (optional)", key="sb_brand", placeholder="Acme Inc.")
    st.selectbox("Brand voice", list(VOICES), key="sb_voice",
                 format_func=lambda v: VOICES[v]["label"])
    st.text_input("Image style", key="sb_style",
                  value="modern flat illustration")
    if st.button("🧹 Reset session"):
        for k in ("memory", "graph", "drafts", "research", "chat"):
            st.session_state.pop(k, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_dash, tab_research, tab_export = st.tabs(
    ["💬 Chat", "📝 Content Dashboard", "🔍 Research", "📤 Export"])

# -- Chat -------------------------------------------------------------------
with tab_chat:
    st.subheader("Talk to ContentBlitz")
    st.caption("Multi-turn conversation — context is preserved. "
               "Try: *research email marketing trends*, then *turn that into "
               "a LinkedIn post*, then *make it punchier*.")
    for msg in S.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask for research, a blog, a post, an image…"):
        brief = _brief_from_sidebar()
        if not brief["topic"]:
            # Let the chat itself carry the topic when the sidebar is empty.
            brief["topic"] = prompt[:80]
        MEM.add_turn("user", prompt)
        S.chat.append({"role": "user", "content": prompt})
        with st.spinner("ContentBlitz is working…"):
            try:
                out = _run(S.graph.ainvoke({
                    "user_input": prompt,
                    "brief": {**brief, **MEM.brief},
                    "drafts": dict(S.drafts),
                    "research": S.research or {},
                }))
            except Exception as exc:  # never crash the UI
                out = {"answer": f"⚠️ Something went wrong: {exc}"}
        answer = out.get("answer", "")
        for kind, res in (out.get("drafts") or {}).items():
            if isinstance(res, dict) and res.get("draft"):
                _store_draft(kind, res)
        if out.get("research"):
            S.research = out["research"]
        MEM.add_turn("assistant", answer[:2000])
        S.chat.append({"role": "assistant", "content": answer})
        st.rerun()

# -- Dashboard --------------------------------------------------------------
with tab_dash:
    st.subheader("Content Dashboard")
    st.caption("Run the full pipeline, then preview AND edit every draft "
               "before publishing.")
    with st.form("pipeline_form"):
        c1, c2 = st.columns(2)
        topic = c1.text_input("Topic*", value=st.session_state.get("sb_topic", ""))
        audience = c2.text_input("Audience", value=st.session_state.get("sb_audience", ""))
        fmts = st.multiselect("Formats",
                              ["blog", "linkedin", "image", "strategy"],
                              default=["blog", "linkedin"])
        do_research = st.checkbox("Research-first (recommended)", value=True)
        go = st.form_submit_button("🚀 Generate content")
    if go:
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            brief = {**_brief_from_sidebar(), "topic": topic.strip(),
                     "audience": audience.strip()}
            MEM.update_brief(**brief)
            with st.spinner("Running the agent pipeline…"):
                try:
                    if do_research:
                        pack = _run(run_research_first(
                            brief, MEM, tuple(fmts)))
                        S.research = pack["research"]
                        for kind, res in pack["outputs"].items():
                            _store_draft(kind, res)
                    else:
                        runners = {"blog": seo_blog_writer,
                                   "linkedin": linkedin_writer,
                                   "image": image_agent,
                                   "strategy": content_strategist}
                        for fmt in fmts:
                            fn = runners[fmt]
                            res = _run(fn(brief, S.research, MEM)
                                       if fmt != "image"
                                       else fn(brief, MEM))
                            _store_draft(fmt, res)
                except Exception as exc:
                    st.error(f"⚠️ Pipeline error: {exc}")
            st.rerun()

    if S.research:
        with st.expander(
                f"🔍 Research pack ({S.research.get('provider', '?')}, "
                f"{S.research.get('citation_count', 0)} sources)"):
            st.markdown(S.research.get("summary", ""))
            for r in S.research.get("results", [])[:8]:
                st.markdown(f"- [{r['title']}]({r['url']}) _({r['source']})_")

    if not S.drafts:
        st.info("No drafts yet — run the pipeline above or ask in 💬 Chat.")
    for kind, res in S.drafts.items():
        if not isinstance(res, dict) or not res.get("draft"):
            continue
        title = {"blog": "✍️ SEO Blog", "linkedin": "💼 LinkedIn Post",
                 "image": "🎨 Visual", "strategy": "🗺️ Strategy"}.get(
                     kind, kind)
        with st.expander(f"{title}  ·  {_quality_badge(res.get('quality'))}",
                          expanded=(kind == "blog")):
            if kind == "image":
                if res.get("status") == "generated" and res.get("image_url"):
                    st.image(res["image_url"], caption=f"Generated image ({res['model']})")
                else:
                    st.warning(res.get("note", "Placeholder image."))
                    st.caption(f"Planned prompt: {res.get('prompt', '')[:400]}")
                continue
            st.markdown(res["draft"])
            if res.get("seo"):
                st.caption(f"SEO keywords: {', '.join(res['seo']['keywords'][:8])}")
                st.caption(f"Meta: {res['seo']['meta_description']}")
            if res.get("issues_fixed"):
                st.caption("Auto-fixed: " + "; ".join(res["issues_fixed"]))
            edit_key = f"edit_{kind}"
            new_text = st.text_area("Edit before publishing", res["draft"],
                                    height=260, key=edit_key)
            if st.button(f"💾 Save edits ({kind})", key=f"save_{kind}"):
                checked = validate_content(new_text, kind,
                                           _brief_from_sidebar().get("brand_voice"))
                res["draft"] = checked["enhanced"]
                res["quality"] = checked["score"]
                if checked["issues"]:
                    st.caption("Fixed on save: " + "; ".join(checked["issues"]))
                MEM.remember_output(kind, res["draft"])
                st.success(f"Saved — new score "
                           f"{checked['score']['score']}/100 "
                           f"(grade {checked['score']['grade']}).")
                st.rerun()

# -- Research ----------------------------------------------------------------
with tab_research:
    st.subheader("Deep Research")
    st.caption("Tavily preferred → SerpApi fallback → labelled demo data. "
               "Every result carries its source.")
    q = st.text_input("Research query",
                      placeholder="e.g. B2B content marketing benchmarks 2026")
    if st.button("🔍 Research") and q.strip():
        with st.spinner("Researching…"):
            try:
                pack = _run(deep_research_agent(
                    {"topic": q.strip()}, MEM))
            except Exception as exc:
                st.error(f"⚠️ Research error: {exc}")
                pack = None
        if pack:
            S.research = pack
            MEM.remember_research(pack["results"])
            st.success(f"{pack['citation_count']} sources via {pack['provider']}.")
            st.markdown("### Summary")
            st.markdown(pack["summary"])
            st.markdown("### Results")
            for r in pack["results"]:
                st.markdown(f"- [{r['title']}]({r['url']}) _({r['source']})_\n"
                            f"  {r['snippet'][:220]}")

# -- Export ------------------------------------------------------------------
with tab_export:
    st.subheader("Export drafts")
    kinds = [k for k, r in S.drafts.items()
             if isinstance(r, dict) and r.get("draft")]
    if not kinds:
        st.info("Nothing to export yet — generate drafts first.")
    else:
        pick = st.multiselect("Drafts to export", kinds, default=kinds)
        fmt = st.radio("Format", ["Markdown", "HTML", "PDF"],
                       horizontal=True)
        if pick and st.button("⬇️ Prepare download"):
            bodies = []
            for k in pick:
                bodies.append(f"# {k.title()}\n\n{S.drafts[k]['draft']}")
            combined = "\n\n---\n\n".join(bodies)
            if fmt == "Markdown":
                st.download_button("Download .md", combined,
                                   file_name="contentblitz.md",
                                   mime="text/markdown")
            elif fmt == "HTML":
                doc = export_html({k: S.drafts[k] for k in pick})
                st.download_button("Download .html", doc,
                                   file_name="contentblitz.html",
                                   mime="text/html")
            else:
                pdf_bytes = export_pdf(combined)
                st.download_button("Download .pdf", pdf_bytes,
                                   file_name="contentblitz.pdf",
                                   mime="application/pdf")
