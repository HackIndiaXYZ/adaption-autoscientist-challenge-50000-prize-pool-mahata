# eval/ — baseline vs. BanglaBridge

This folder proves the **eligibility gate**: a measurable % improvement over Adaption's
baseline on their held-out Language test set. Make it dead simple for a judge to verify.

## Methodology
- **Test set:** Adaption's held-out Language test set (never seen in training).
- **Metric:** `TODO` (match whatever Adaption scores on — accuracy / BLEU / win-rate / etc.)
- **Procedure:** run baseline and BanglaBridge on the *same* inputs, same decoding params,
  same metric. Report both + the delta.

## Headline result (fill in)

| System | Metric | Score |
| --- | --- | --- |
| Adaption baseline | `TODO` | `TODO` |
| **BanglaBridge** | `TODO` | **`TODO`** |
| **Improvement** | | **`TODO %`** |

## Files (add as you go)
- `run_eval.py` — reproducible eval script (baseline + ours) — `TODO`
- `results.json` — raw scores — `TODO`
- `examples.md` — side-by-side outputs where BanglaBridge clearly wins (great for the demo
  and social posts)

## Integrity
- Confirm zero overlap between eval inputs and training data.
- Use identical prompts/decoding for both systems — the only variable is the model.
