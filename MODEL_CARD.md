# Model Card — BanglaBridge-Instruct

<!-- This doubles as the README on the Hugging Face model repo. Keep it in sync. -->

## Model summary
- **Model:** BanglaBridge-Instruct
- **Base model:** `TODO (baseline provided by Adaption for the Language category)`
- **Adaptation method:** instruction fine-tuning via Adaption AutoScientist (data↔recipe co-optimization). LoRA/QLoRA adapters `TODO`.
- **Language(s):** Bengali (code-mixed / romanized "Banglish") + English
- **License:** `TODO (e.g. apache-2.0 / cc-by-4.0 — must match base model terms)`
- **Developed by:** Team MAHATA — Adaption AutoScientist Challenge × HackIndia
- **Repo:** https://github.com/HackIndiaXYZ/adaption-autoscientist-challenge-50000-prize-pool-mahata

## Intended use
Instruction-following, Q&A, and generation in code-mixed / romanized Bengali — chat
assistants, content tools, and support bots serving Bengali speakers who type in the
romanized register.

**Out of scope:** high-stakes medical/legal/financial advice; safety-critical decisions.

## Results (the headline)

| Metric | Baseline (Adaption) | BanglaBridge | Δ improvement |
| --- | --- | --- | --- |
| Adaption held-out Language test | `TODO` | `TODO` | **`TODO %`** |
| `TODO secondary metric` | `TODO` | `TODO` | `TODO` |

Evaluation methodology and scripts: see `eval/`. Improvement over baseline is the
challenge's eligibility gate — this table is the single most important artifact.

## Training data
Original adapted Banglish instruction dataset — sourcing, cleaning, dedup, and licensing
documented in `DATASET_CARD.md` and `data/`. Released openly alongside the model.

## Training procedure
- Platform: Adaption AutoScientist (free compute)
- Recipe / hyperparameters / run logs: `training/`
- Reproducibility: configs committed; run link in `README.md`.

## Limitations & risks
- Register-specialized: may underperform generic models on formal native-script Bengali.
- Inherits base-model biases; code-mixed data may carry informal/social-media biases.
- `TODO — fill after eval (failure cases, hallucination rate, etc.)`

## Citation
```
@misc{banglabridge2026,
  title  = {BanglaBridge-Instruct: a domain-adapted LLM for code-mixed Bengali},
  author = {Team MAHATA},
  year   = {2026},
  note   = {Adaption AutoScientist Challenge x HackIndia},
  url    = {https://github.com/HackIndiaXYZ/adaption-autoscientist-challenge-50000-prize-pool-mahata}
}
```
