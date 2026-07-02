# Sources, Provenance & Methodology

> **One-line claim:** every pair in this dataset is authored by us or produced by a
> deterministic transformation of our own templates. **Nothing is scraped.** The set is
> therefore fully safe to redistribute (CC-BY-4.0) on Hugging Face and Kaggle.

## Source breakdown

| Source                | What it is                                              | License | Redistributable |
|-----------------------|--------------------------------------------------------|---------|-----------------|
| Authored GOLD         | Hand-written natural Banglish instruction pairs         | Ours    | ✅ |
| Curated BANKS         | Hand-written short pairs (sentiment, translate, rewrite, grammar, intent, extract, math, QA, safety, summarize, reasoning, native) with guaranteed-correct outputs | Ours | ✅ |
| SLOTTED templates     | Our templates + safe slot substitution (fabricated names only) | Ours | ✅ |
| Spelling augmentation | Deterministic meaning-preserving respellings of our own train rows | Ours | ✅ |

No web scraping, no third-party corpora, no user data. All names, phone numbers, emails,
PIN codes and addresses are **fabricated placeholders**.

## The spelling-augmentation method (the robustness multiplier)

Banglish has **no standard orthography**. The same word is typed many ways:

| meaning       | variants seen in the wild            |
|---------------|--------------------------------------|
| "is/are"      | `ache` · `ase` · `achhe`             |
| "how"         | `kivabe` · `kibhabe` · `kemne` · `kmne` |
| "good"        | `bhalo` · `valo` · `vlo`             |
| "one"         | `ekta` · `akta` · `1ta`              |
| "my"          | `amar` · `amr`                       |

Off-the-shelf models overfit to one spelling and break on the rest. We fix this at the
data level:

1. For each **train** row (romanized/mixed only — never native), tokenize the *instruction*.
2. Replace eligible tokens with a **meaning-preserving** variant (probability 0.6 each).
   We never touch person/formality markers (e.g. never `tumi`↔`tui`).
3. Keep the **output identical** — only the surface spelling of the input changes.
4. Deduplicate against every existing instruction; tag `source += "+aug:spelling"`,
   `augmented: true`.

**Held-out is left canonical** (single clean spelling) so evaluation is honest and
leakage-free. Deterministic under `seed=42`. Toggle with `--aug-variants N` (0 disables).

## Design decisions

- **Input register:** majority romanized code-mix; a native-script + mixed-script minority
  for robustness — mirrors how people actually type.
- **Output register:** replies **in the user's register** by default (the thing off-the-shelf
  models fail at). Exceptions: translate/explain tasks, where the target language is explicit.
- **Correctness:** bank outputs are hand-written alongside inputs, so we never rely on
  auto-translation (which would introduce errors). Math/QA/reasoning answers are verified.
- **Safety:** harmful requests get graceful in-register refusals; safety-critical prompts
  (chest pain, self-harm, scams/blackmail) escalate to real helplines.

## Scaling beyond the seed (for the full submission run)

This 532-pair set is sized to **prove lift** fast. To scale to thousands while staying
redistribution-safe:

1. **LLM paraphrase augmentation** — prompt an LLM to paraphrase our GOLD instructions
   *within the same Banglish register* (keep outputs fixed or regenerate + human-spot-check).
   Because the seeds are ours, the paraphrases are ours to release.
2. **More slot dictionaries** — extend `NAMES`/cities/foods/times for the SLOTTED templates.
3. **Wider spelling map** — grow `SPELLING_VARIANTS` and/or `--aug-variants 2`.
4. **Adaptive Data pass** — run the whole set through Adaption's platform
   (`data/adaptive_pipeline.py`) to quality-optimize before training.

Keep provenance for anything added here so the "safe to redistribute" claim stays true.
