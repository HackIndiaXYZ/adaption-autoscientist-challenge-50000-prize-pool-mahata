# data/ — BanglaBridge Banglish instruction dataset

The originality axis of the submission. Everything here is **original / synthetic** and
**safe to redistribute** — see `sources.md` for full provenance.

## Build it
```bash
python build_dataset.py               # deterministic (seed=42), pure stdlib, no network
python build_dataset.py --stats-only  # print the distribution without writing files
python build_dataset.py --aug-variants 2   # more spelling variants per train row
```

## What gets written
| File                          | Contents                                                  |
|-------------------------------|-----------------------------------------------------------|
| `banglish_instructions.jsonl` | full dataset, rich schema (one JSON object per line)      |
| `banglish_instructions.csv`   | flat `original_prompt,response` for the Adaption pipeline |
| `train.jsonl` / `heldout.jsonl` | 85/15 split, **no leakage** (heldout kept canonical)    |
| `sample.jsonl`                | first 15 rows, for quick review                           |

## Schema
```json
{"id": "…", "instruction": "kal weather kemon thakbe bol to?", "input": "",
 "output": "…", "task_type": "qa", "domain": "daily_life",
 "script": "romanized", "source": "gold", "augmented": false, "split": "train"}
```

## How it's made (`build_dataset.py`)
1. **GOLD** — hand-authored natural Banglish pairs across 20+ task types.
2. **BANKS** — curated short pairs with guaranteed-correct outputs (sentiment, translate,
   rewrite, grammar, intent, extract, math, factual QA, safety, summarize, reasoning, native).
3. **SLOTTED** — our templates + safe slot substitution (fabricated names) for volume.
4. **Split** — seeded shuffle → train/held-out, deduped on normalized instruction.
5. **Spelling augmentation** — deterministic meaning-preserving respellings of *train*
   rows only (the Banglish robustness multiplier). See `sources.md`.

## Feed it to Adaption
`adaptive_pipeline.py` ingests `banglish_instructions.csv`
(`column_mapping={"prompt": "original_prompt"}`), runs the adapt pass, and reports the
before→after quality lift. Start on a small `max_rows` slice to gauge credit cost.

> Full corpus is small enough to keep in git. If it grows large, host on HF/Kaggle and
> keep only scripts + `sample.jsonl` here.
