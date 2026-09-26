# ✍️ ContentBlitz — AI Content Marketing Agent

Interview Kickstart capstone, **agent 2**: a LangGraph multi-agent system for
content marketing, with a usable Streamlit content-creation UI.

> **Build status: Phase 1 scaffold.** The full agent pipeline is being built
> exactly to the course spec in Phase 2.

## Project layout

```
contentblitz-ai-content-marketing/
├── app.py                 # Streamlit UI (shell in Phase 1)
├── contentblitz/          # Agent package (pipeline lands in Phase 2)
├── tests/                 # pytest suite
├── test_data/             # Sample briefs + expected behavior (Phase 2)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

All secrets come from **environment variables only** — never commit keys:

| Variable         | Purpose                              |
|------------------|--------------------------------------|
| `OPENAI_API_KEY` | Chat/completion model for the agents |

Without keys the app degrades gracefully (offline/demo behavior).

## Testing

```bash
pytest
```
