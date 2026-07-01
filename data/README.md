# data/ — Banglish instruction dataset build

The originality axis of the submission lives here. Document sourcing → cleaning → split so
a judge can reproduce the dataset exactly.

## Pipeline
1. **Source** — collect/synthesize code-mixed Bengali prompt→response pairs. Record every
   source + license in `DATASET_CARD.md`.
2. **Ingest** — load into Adaption's Adaptive Data platform.
3. **Clean** — dedup, normalize romanization, PII scrub, quality filter.
4. **Adapt** — Adaptive Data quality/eval passes.
5. **Split** — train vs. held-out. Never leak held-out into training.

## Files (add as you build)
- `build_dataset.py` — `TODO`
- `clean.py` — `TODO`
- `sources.md` — one row per source: name, URL, license, count
- `sample.jsonl` — a few example rows (for reviewers)

## Format
```json
{"instruction": "kal weather kemon thakbe bosht?", "input": "", "output": "..."}
```

> Do NOT commit the full raw corpus here if it's large — host it on HF/Kaggle and link it.
> Keep only scripts + a small `sample.jsonl` in git.
