# ✍️ ContentBlitz — AI Content Marketing Agent

Interview Kickstart capstone, **agent 2**: a LangGraph multi-agent system
that turns a topic into researched, SEO-optimized, brand-consistent content
— blog articles, LinkedIn posts, visuals, and strategy docs — through a
conversational Streamlit UI.

## What it does

- **Research-first workflow:** every research-worthy request runs deep web
  research *before* any writing, then repurposes the same research pack
  into multiple formats.
- **Six specialized agents** behind an intelligent Query Handler.
- **Multi-turn conversation** with persistent memory (brief, drafts,
  research, last 20 turns) — refine drafts iteratively
  (*"make it punchier"*), repurpose across formats (*"turn that into a
  LinkedIn post"*).
- **Content dashboard:** preview **and edit** every draft before
  publishing; edits are re-validated and re-scored on save.
- **Export** drafts as Markdown, HTML, or PDF.

## The six agents

| Agent | Role |
|---|---|
| **Query Handler** | Intelligent LLM router (structured JSON decision). Runs on the initial request **and** again after research completes (post-research routing). Deterministic keyword fallback when offline. |
| **Deep Research** | Fans out 2–3 angled sub-queries; compiles a cited research pack. Tavily → SerpApi → labelled mock. |
| **SEO Blog Writer** | Long-form articles through the SEO engine + validation pipeline. |
| **LinkedIn Post Writer** | Platform-native posts: hook, short lines, question CTA, hashtags. |
| **Image Agent** | Optimizes the visual prompt (LLM art-direction polish); Pollinations.ai (free, keyless) → DALL-E 3 → DALL-E 2 → clearly labelled placeholder. |
| **Content Strategist** | Organizes research into positioning, key messages, outline, repurposing plan, distribution. |

## Architecture

```
                    ┌─────────────────┐
                    │  Query Handler  │  (initial routing)
                    │  LLM router     │
                    └────────┬────────┘
                             │ needs_research?
                    ┌────────▼────────┐
                    │  Deep Research  │  Tavily → SerpApi → mock
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │  Query Handler  │  (post-research routing)
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌───────────┐  ┌────────────┐
        │ SEO Blog │  │ LinkedIn  │  │ Strategist │  (+ combo / refine / series / image)
        └────┬─────┘  └─────┬─────┘  └─────┬──────┘
             └──────────────┼──────────────┘
                            ▼
              SEO engine → validation/enhancement → citations
```

**State** (`BlitzState`): user input, brief, router decision, research
pack, pending formats, drafts, answer, plus an internal `_researched`
flag that drives the research-first leg. Memory (`ConversationMemory`)
persists across turns: turns, working brief, last outputs, last research.

### Trade-offs

- **LangGraph over plain chains:** the router genuinely needs to run
  twice (before and after research) with conditional edges — a linear
  chain can't express post-research routing cleanly. Cost: more
  machinery than a simple pipeline.
- **LLM router + keyword fallback (not pure LLM, not pure rules):** the
  LLM makes nuanced decisions when a key exists; the deterministic
  fallback keeps the app 100% usable offline (e.g. Streamlit Cloud
  without secrets). Cost: two routing paths to maintain and test.
- **Research-first default:** better-grounded content, but adds latency
  and API cost per request. The dashboard lets you disable it.
- **Synchronous thread-wrapped I/O** (`asyncio.to_thread`) for the
  blocking `requests`/OpenAI clients instead of async HTTP libraries —
  simpler dependency footprint, fine at this scale.
- **Heuristic quality/voice scoring** (Flesch, structure, keyword
  density, voice rules) rather than LLM-as-judge: deterministic,
  instant, free — but dumber than an LLM critic. Good enough for
  draft-level feedback; not a substitute for human review.
- **fpdf2 core fonts for PDF export:** zero font-file hassle, but
  latin-1 only — non-latin scripts are replaced, not rendered.

## Integrations & fallbacks

| Capability | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| LLM (router, writers, strategist) | OpenAI `gpt-4o-mini` via LangChain | — | Deterministic offline templates (labelled) |
| Web research | Tavily | SerpApi | Labelled mock pack |
| Image generation | Pollinations.ai (free, keyless) | DALL-E 3 | DALL-E 2, then labelled placeholder (never shown as real) |

**Rules:**
- A missing key or failed request **never** crashes the pipeline — every
  adapter returns `None` on any exception and the next fallback engages.
- Every result carries its `source`/`provider` label; the UI shows
  `offline mode` / `demo data` / `labelled placeholder` honestly.
- Demo citations render as `_(demo)_`; live ones as `_(live)_`.
- A free `/v1/models` probe plus a keyless default (`check_image_support()`)
  reports image capability: Pollinations.ai is always on (no key), DALL-E
  availability depends on the OpenAI key.
- No secret ever appears in the UI, logs, README, or git history.

## Validation process

Every draft passes through the **validation & enhancement pipeline**
(`contentblitz/quality.py`):

1. **Auto-fixes** — collapse 3+ blank lines, trim over-long LinkedIn
   drafts, fix dangling trailing colons.
2. **Quality score (0–100)** — length (25), structure/headers (25),
   readability/Flesch (25), engagement/CTA (25) → grade A–F with
   actionable suggestions.
3. **Voice check** — heuristic brand-voice consistency (professional /
   friendly / bold / technical): emoji, hype-word, and sentence-length
   rules; drift is reported as an issue.
4. **SEO engine** — keyword extraction (research-grounded), meta
   description (≤160 chars), slug, header optimization, keyword density.
5. **Citations** appended with honest live/demo labels.

**Iterative refinement** (`refine_content`): up to N rounds, keeps the
best-scoring version. **Series generation** (`generate_series`):
strategist defines the arc, each part is drafted and scored.

## Service Comparison Analysis

Cost-benefit, benchmarks, and recommendations by use case/budget.
(Prices are public list prices as of 2026; verify before budgeting.)

### LLM providers (router + writers)

| Service | Approx. cost | Quality | Notes |
|---|---|---|---|
| OpenAI `gpt-4o-mini` | ~$0.15 / $0.60 per 1M tokens (in/out) | Strong | **Default.** Best price/quality for marketing copy; structured JSON routing is reliable. |
| OpenAI `gpt-4o` | ~$2.50 / $10 per 1M tokens | Best | ~15× the cost of mini for marginally better drafts — not worth it for this workload. |
| Anthropic Claude Haiku | ~$0.25 / $1.25 per 1M tokens | Strong | Good alternative; slightly better tone control, slightly pricier. |
| Local / Ollama | $0 (own hardware) | Weaker | Zero marginal cost, but routing JSON reliability drops — keep the keyword fallback as primary. |

**Benchmarks (this codebase):** a research-first blog run makes ~6–8 LLM
calls (router ×2, research summary, writer, series arc optional). At
gpt-4o-mini pricing that's **well under $0.05 per full pipeline run**.
Offline mode costs $0 and still produces structured, scored drafts.

### Research providers

| Service | Approx. cost | Quality | Notes |
|---|---|---|---|
| Tavily | Free tier + ~$0.008/search paid | Best | **Preferred.** Search purpose-built for AI: clean snippets, `include_answer`. |
| SerpApi | Free tier (100/mo) + from ~$50/mo | Good | **Fallback.** Google results; noisier, needs snippet trimming. Fine as backup. |
| Mock (built-in) | $0 | Demo only | Deterministic, labelled. For UI dev and CI — never for real briefs. |

### Image providers

| Service | Approx. cost | Quality | Notes |
|---|---|---|---|
| Pollinations.ai (Flux) | $0 (no key, no signup) | Good | **Primary.** Keyless free tier; generations are private by default. |
| DALL-E 3 | ~$0.04/image (1024²) | Best | **Paid fallback.** Good prompt adherence for marketing visuals. |
| DALL-E 2 | ~$0.02/image | Good | **Paid fallback.** Cheaper, weaker composition. |
| Labelled placeholder | $0 | N/A | Honest "not generated" card with the planned prompt — better than a fake image. |

### Recommendations

- **Bootstrapped / course demo:** all-offline mode ($0). Everything
  runs; outputs are templates with real structure, SEO artifacts, and
  scores.
- **Solo marketer (recommended):** `gpt-4o-mini` + Tavily free tier +
  Pollinations image gen (free). Full pipeline ≈ **<$5/month** at moderate
  volume, and images cost nothing.
- **Agency / high volume:** `gpt-4o-mini` + Tavily paid + DALL-E 3.
  Expected **$20–60/month** for hundreds of runs — the research-first
  workflow is the main cost driver (2–3 searches per run).
- **Quality-max:** swap mini → `gpt-4o` for the writers only (keep mini
  for the router). ~10–15× LLM cost for a modest quality lift.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Environment variables (all optional)

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM routing/writing + DALL-E image fallback (Pollinations needs no key) |
| `TAVILY_API_KEY` | Web research (preferred) |
| `SERPAPI_API_KEY` | Web research fallback |

Without keys the app runs fully offline with labelled demo behavior.

## Testing

```bash
# Full suite (hermetic — no keys, no network)
.venv/bin/python -m pytest tests/ -p no:cacheprovider --cov=contentblitz

# Live integration probes (needs real keys; cheap — no images generated)
OPENAI_API_KEY=... TAVILY_API_KEY=... SERPAPI_API_KEY=... \
  .venv/bin/python -m pytest tests/test_live_integrations.py -p no:cacheprovider
```

**Quality gate (enforced before every push):** 101 tests green, **98%
coverage** (target ≥90%), every agent route exercised end to end,
Streamlit health endpoint 200 with no traceback.

`test_data/sample_briefs.md` has 8 realistic briefs (blog, LinkedIn,
image, strategy, multi-format, series, multi-turn refinement,
research-only); `test_data/expected_behavior.md` documents offline/live
behavior, invariants, and known limitations.

## Project layout

```
├── app.py                  # Streamlit UI: Chat / Dashboard / Research / Export
├── contentblitz/
│   ├── agents.py           # 6 agents: router, research, blog, linkedin, image, strategist
│   ├── graph.py            # LangGraph orchestration (router runs before + after research)
│   ├── workflows.py        # research-first, series, iterative refinement
│   ├── tools.py            # Tavily→SerpApi→mock, Pollinations→DALL-E 3→2→placeholder
│   ├── quality.py          # scoring, SEO engine, validation/enhancement pipeline
│   ├── brand.py            # brand voices + consistency checks
│   ├── memory.py           # conversation memory
│   ├── exports.py          # Markdown/HTML/PDF export helpers
│   └── config.py           # env-only keys, proxy sanitization
├── tests/                  # pytest suite (hermetic + live probes)
└── test_data/              # sample briefs + expected behavior
```

## Known limitations

- **DALL-E not exercised against the real API here** (no key in this
  environment) — the DALL-E 3→2 legs are verified with a stubbed
  OpenAI client. The Pollinations leg is verified against the live
  endpoint in `test_live_integrations.py` only when run manually.
  Real availability depends on the key's model access;
  use `check_image_support()` before promising image generation.
- **Live LLM/research paths** verified via stubbed transports only;
  run `tests/test_live_integrations.py` with real keys for a live check.
- **PDF export** uses fpdf2 core fonts (latin-1); non-latin scripts are
  replaced, not rendered.
- Heuristic quality/voice scoring is draft-level feedback, not a
  substitute for human editorial review.
