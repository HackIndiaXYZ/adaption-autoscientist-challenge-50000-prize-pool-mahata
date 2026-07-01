# training/ — AutoScientist recipe & run logs

Depth of AutoScientist usage is a judging axis. Show the *loop*, not a one-shot fine-tune.

## What to capture
- **Baseline** — exact model + config Adaption provides for Language.
- **Recipe** — hyperparameters, LoRA/QLoRA settings, epochs, LR schedule.
- **Co-optimization** — each AutoScientist iteration: what changed in the data or recipe,
  and the held-out metric after it. A short table of iterations is very persuasive.
- **Run link** — the AutoScientist run URL (also in root `README.md`).

## Iteration log (fill in)

| Iter | Change (data / recipe) | Held-out metric | Δ vs. prev |
| --- | --- | --- | --- |
| 0 | baseline (no adaptation) | `TODO` | — |
| 1 | `TODO` | `TODO` | `TODO` |
| 2 | `TODO` | `TODO` | `TODO` |

## Files (add as you go)
- `recipe.yaml` / config export from AutoScientist — `TODO`
- `run_notes.md` — chronological log of what you tried and why
