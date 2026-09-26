"""ContentBlitz — AI Content Marketing Agent. Streamlit UI.

Content-creation workspace: brief in, multi-agent pipeline out.
Full pipeline UI lands in Phase 2 with the course spec.

Run:  streamlit run app.py
Keys via environment (all optional — the app degrades gracefully):
    OPENAI_API_KEY   Chat/completion model for the agent pipeline
"""
from __future__ import annotations

import streamlit as st

import contentblitz

st.set_page_config(page_title="ContentBlitz — AI Content Marketing", page_icon="✍️", layout="wide")
st.title("✍️ ContentBlitz — AI Content Marketing Agent")
st.caption(f"Multi-agent content pipeline · v{contentblitz.__version__}")

with st.sidebar:
    st.header("Status")
    st.write("🚧 Build status: Phase 1 scaffold")
    st.caption("Agent pipeline arrives in Phase 2, built to the course spec.")

st.info(
    "Scaffold is up. The full content-creation pipeline "
    "(brief → research → draft → edit → repurpose) lands in Phase 2."
)

with st.form("brief_form"):
    st.subheader("Content brief (placeholder)")
    topic = st.text_input("Topic", placeholder="e.g. Launching a new SaaS analytics feature")
    audience = st.text_input("Audience", placeholder="e.g. B2B product managers")
    submitted = st.form_submit_button("Generate (coming in Phase 2)", disabled=True)
    if submitted:
        st.write(topic, audience)  # pragma: no cover
