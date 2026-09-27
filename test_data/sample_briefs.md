# ContentBlitz — Sample Briefs

Realistic briefs used for manual and automated verification. Each brief
exercises a different agent route end to end.

## 1. Blog brief (SEO Blog Writer, research-first)

- **Topic:** Email marketing benchmarks for SaaS in 2026
- **Audience:** SaaS founders and growth marketers
- **Brand voice:** professional
- **Prompt:** `write an SEO blog post about email marketing benchmarks for SaaS in 2026`
- **Expected route:** Query Handler (intent=blog, needs_research) → Deep
  Research → post-research routing → SEO Blog Writer
- **Expect:** H1 + 4–6 H2s + FAQ + conclusion, citations section, SEO
  keywords/meta/slug attached, quality score ≥ 60.

## 2. LinkedIn brief (LinkedIn Post Writer)

- **Topic:** Our product launch
- **Audience:** B2B product managers
- **Brand voice:** bold
- **Prompt:** `write a LinkedIn post about our launch`
- **Expected route:** Query Handler (intent=linkedin, no research) →
  LinkedIn Post Writer
- **Expect:** 120–220 words, hook first line, question CTA, 3–5 hashtags,
  bold-voice consistency score high.

## 3. Image brief (Image Generation Agent)

- **Topic:** Launch campaign hero visual
- **Audience:** social media followers
- **Image style:** modern flat illustration
- **Mood:** optimistic and professional
- **Prompt:** `generate an image for the launch visual`
- **Expected route:** Query Handler (intent=image) → Image Agent
- **Expect (no key):** status=placeholder, explicit PLACEHOLDER note, never
  presented as a real image. **Expect (with key):** DALL-E 3 attempt first,
  then DALL-E 2, then placeholder.

## 4. Strategy brief (Content Strategist)

- **Topic:** Q4 content pillars for a dev-tools startup
- **Audience:** developer marketers
- **Brand voice:** technical
- **Prompt:** `build a content strategy for Q4`
- **Expected route:** Query Handler (intent=strategy) → Content Strategist
- **Expect:** positioning, 3–5 key messages, outline, repurposing plan,
  distribution notes.

## 5. Research-first multi-format brief (combination)

- **Topic:** Customer onboarding best practices
- **Audience:** SaaS founders
- **Prompt:** `write a blog and a LinkedIn post about onboarding`
- **Expected route:** Query Handler (intent=combo, needs_research) → Deep
  Research → post-research routing → combo node (blog + LinkedIn)
- **Expect:** both drafts produced, grounded in the same research pack.

## 6. Campaign / series brief

- **Topic:** Product launch
- **Audience:** B2B buyers
- **Prompt:** `plan a 5-part series for our launch campaign`
- **Expected route:** Query Handler (intent=series, needs_research) → Deep
  Research → post-research routing → series node
- **Expect:** 5 scored parts (series_part_1 … series_part_5) in drafts.

## 7. Multi-turn refinement (conversation memory)

1. `write a LinkedIn post about our launch` → draft stored in memory
2. `make it shorter and punchier` → intent=refine, target_kind=linkedin,
   improved (or equal) quality score, ≤ 2 rounds
3. `turn that into a blog post` → reuses prior research, produces a blog

## 8. Research-only brief

- **Prompt:** `research B2B content marketing trends and statistics`
- **Expected route:** Query Handler (intent=research) → Deep Research →
  post-research routing → Content Strategist (organizes findings)
- **Expect:** cited research pack; offline mode labels every source `demo`.
