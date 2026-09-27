# ContentBlitz — Expected Behavior

How the system should behave in each mode. Verified by the pytest suite
(`tests/`) and the Streamlit smoke check.

## Offline mode (no API keys — the default on a fresh clone)

- **Router:** deterministic keyword fallback. LLM routing engages only when
  `OPENAI_API_KEY` is set.
- **Research:** every provider unavailable → labelled mock pack
  (`provider: "mock"`, each result `source: "mock"`). Citations render as
  `_(demo)_` — demo citations are never presented as live.
- **Writers:** deterministic offline templates, tagged
  `_(Drafted in offline mode — set OPENAI_API_KEY for AI-polished copy.)_`
- **Images:** `status: "placeholder"` with an explicit PLACEHOLDER note.
  The UI shows a warning, never a fake image.
- **App:** sidebar shows `offline mode` / `demo data` / `labelled
  placeholder`. Everything still runs end to end.

## Live mode (keys in environment)

- **Router:** LLM returns structured JSON
  `{intent, needs_research, formats, confidence, reasoning}`; malformed
  JSON falls back to the keyword router (tested).
- **Research:** Tavily first; on failure SerpApi; on failure labelled mock.
  Provider is recorded on every pack (`provider: "tavily"|"serpapi"|"mock"`).
- **Writers:** LLM drafts → SEO engine (keywords, meta ≤160 chars, slug,
  header optimization, keyword density) → validation/enhancement pipeline
  (quality score, voice check, auto-fixes) → citations appended.
- **Images:** DALL-E 3 → DALL-E 2 → labelled placeholder. The free
  `/v1/models` probe (`check_image_support`) reports capability without
  spending credits.
- **Errors:** any provider exception → `None` → next fallback. The UI
  catches per-pipeline exceptions and shows a warning, never a traceback.

## Invariants (must always hold)

1. A placeholder image is NEVER presented as a real generated image.
2. Demo citations are always labelled `demo`; live citations `live`.
3. No API key or secret ever appears in the UI, logs, README, or git.
4. Every writer output carries `quality` (score/grade/suggestions),
   `voice_check`, and `issues_fixed`.
5. Conversation memory persists across turns: brief, drafts, research,
   and the last 20 turns.
6. The research-first workflow always runs fresh research for a new
   topic (stale research never suppresses it) — tested in
   `test_new_topic_triggers_fresh_research`.
7. `pytest -p no:cacheprovider` is green with ≥90% coverage before any push.

## Known limitations

- **DALL-E:** not exercised against the real API in this environment (no
  key); the chain is verified with a stubbed OpenAI client. Real
  availability depends on the key's model access — use the sidebar/app
  `check_image_support()` probe before promising image generation.
- **Live research/LLM:** verified via stubbed transports only; run
  `tests/test_live_integrations.py` with real keys for a live check.
- **PDF export:** fpdf2 core fonts only (latin-1); non-latin scripts are
  replaced, not rendered.
