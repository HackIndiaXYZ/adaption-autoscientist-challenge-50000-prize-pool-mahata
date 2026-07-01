# Dataset Card — BanglaBridge Banglish Instruction Set

<!-- This doubles as the README on the Hugging Face + Kaggle dataset repos. -->

## Summary
An original instruction-tuning dataset for **code-mixed / romanized Bengali ("Banglish")**:
prompt→response pairs in the register Bengali speakers actually type online. Built and
adapted with Adaption's Adaptive Data platform. This is the originality axis of the
submission — document it thoroughly.

## Why it's original (not a re-upload)
- `TODO` — describe what makes this set new: the code-mixed/romanized register, task mix,
  cleaning pipeline, or synthesis method. Judges reward originality, so be specific.

## Composition
- **Size:** `TODO` examples
- **Format:** `{ "instruction": ..., "input": ..., "output": ... }` (or chat turns) — `TODO`
- **Register mix:** pure Banglish / transliteration / Bengali-English QA — `TODO` %
- **Task mix:** `TODO` (QA, rewrite, summarize, translate, chat, ...)

## Sourcing
- Sources: `TODO` (list every source + its license/permission)
- Synthetic generation (if any): model + prompts used — `TODO`
- ⚠️ Only use sources you have the right to redistribute. Record licenses per source.

## Cleaning & adaptation pipeline
1. Ingest via Adaptive Data — `TODO`
2. Dedup / normalize romanization — `TODO`
3. Quality + eval passes on-platform — `TODO`
4. Train/held-out split (never leak into eval) — `TODO`

Scripts: `data/`.

## Licensing
- Dataset license: `TODO` (e.g. cc-by-4.0)
- Confirm every upstream source permits redistribution under this license.

## Ethical considerations
- Social-media-sourced text may contain informal/biased/PII content — describe your
  PII scrub and filtering here. `TODO`
