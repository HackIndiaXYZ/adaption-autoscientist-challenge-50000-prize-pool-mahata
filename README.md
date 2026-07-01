# BanglaBridge — a domain-adapted LLM for code-mixed Bengali ("Banglish")

**Adaption AutoScientist Challenge · Category: Language · HackIndia track (Team MAHATA)**

Off-the-shelf LLMs collapse on the romanized, code-switched Bengali that 100M+ people
actually type every day ("ami tomake bhalobashi", "kal office jabo na"). BanglaBridge is
an instruction-tuned model, adapted with Adaption's **Adaptive Data** + **AutoScientist**
loop, that measurably beats the Language baseline on Adaption's held-out test set — and is
released openly on Hugging Face and Kaggle with full reproducibility.

> Status: setup / pre-training. Numbers below are filled in as the AutoScientist runs complete.

---

## Why this wins on the rubric

| Judging axis | How BanglaBridge scores |
| --- | --- |
| **Measurable improvement over baseline** | Generic baselines are *weak* on code-mixed/romanized Bengali → large, unmistakable delta. Target: **≥ X% lift** on Adaption's held-out Language test set. |
| **Dataset quality & originality** | An original, cleaned Banglish instruction set — not a re-upload of an existing corpus. Sourcing + cleaning documented in `data/`. |
| **Real-world impact** | Code-mixed Bengali is how the language is *actually written online*; no mainstream model handles it well. Directly serves 100M+ speakers. |
| **Depth of AutoScientist usage** | Full data↔recipe co-optimization loop, logged in `training/`. Not a one-shot fine-tune. |
| **Open release quality** | Model card, dataset card, eval methodology, and reproducible pipeline all in this repo + mirrored to HF & Kaggle. |

## Links (fill in on release)

- 🤗 Hugging Face model: `TODO`
- 🤗 Hugging Face dataset: `TODO`
- 📦 Kaggle model: `TODO`
- 📦 Kaggle dataset: `TODO`
- 🔬 Adaption AutoScientist run: `TODO`
- 🎬 Live demo: `TODO`
- 📣 LinkedIn post: `TODO` · 𝕏 post (tag @adaption_ai): `TODO`

## Repo layout

```
adaption-autoscientist/
├── README.md                 # this file
├── PROBLEM_STATEMENT.md      # the crisp problem statement (paste into HackIndia)
├── MODEL_CARD.md             # HF/Kaggle model card (fill as you train)
├── DATASET_CARD.md           # dataset card (sourcing, cleaning, licensing)
├── SUBMISSION_CHECKLIST.md   # every step required for prize eligibility
├── data/                     # dataset build scripts + notes
├── training/                 # AutoScientist recipe, configs, run logs
└── eval/                     # baseline-vs-ours evaluation methodology + results
```

## Reproduce in one read

1. `data/` — how the Banglish instruction dataset was sourced, cleaned, and adapted.
2. `training/` — the exact AutoScientist recipe + configs used to fine-tune.
3. `eval/` — how we score against Adaption's baseline on the held-out test set.

## Team

**MAHATA** — HackIndia · Adaption AutoScientist Challenge (Part 1, Language).
