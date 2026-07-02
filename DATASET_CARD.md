# Dataset Card — BanglaBridge Banglish Instruction Set

<!-- This doubles as the README on the Hugging Face + Kaggle dataset repos. -->

## Summary
An **original** instruction-tuning dataset for **code-mixed / romanized Bengali
("Banglish")** — the register 100M+ people actually type online
(e.g. *"kal ki plan? ami free achi"*). Every pair is authored by us or produced by
safe, deterministic transformation of our own templates. **Nothing is scraped**, so the
whole set is free to redistribute on Hugging Face and Kaggle.

This is the originality + real-world-impact axis of the submission: off-the-shelf LLMs
garble Banglish, switch to formal Bengali/English, or ignore the register entirely. This
set teaches a model to **understand Banglish and reply in the user's register.**

## Composition
- **Size:** 532 instruction pairs (335 authored/base + 197 spelling-augmented)
- **Split:** 482 train / 50 held-out (our internal split; the *real* eval is Adaption's
  held-out test set). **Zero normalized-instruction leakage** between splits (verified).
- **Format (JSONL):**
  ```json
  {"id": "…", "instruction": "…", "input": "", "output": "…",
   "task_type": "…", "domain": "…", "script": "romanized|native|mixed",
   "source": "…", "augmented": false, "split": "train|heldout"}
  ```
  A flat `banglish_instructions.csv` (`original_prompt,response`) is also emitted for the
  Adaption Adaptive-Data pipeline.
- **Script mix:** romanized ~96% · native-script ~3% · mixed ~1% (mirrors real usage:
  romanized dominates; native/mixed anchor robustness).
- **Task mix (21 types):** translate, creative, classification, rewrite, grammar-fix,
  intent, math, factual QA, advice, extraction/NER, **safety/refusal**, summarize,
  reasoning, multi-turn, roleplay, code, emotional-support, planning, explanation,
  how-to, mixed.
- **Domains:** daily-life, education, work, relationships, tech, food, health, travel,
  finance, entertainment.

## Why it's original (not a re-upload)
1. **Register.** Native, natural romanized code-mix — not machine-transliterated Bengali.
2. **In-register responses.** Outputs match the user's register (the exact failure mode
   of off-the-shelf models), except translate/explain tasks where the target is explicit.
3. **Spelling-robustness augmentation.** Banglish has *no* standard orthography
   (`ache/ase/achhe`, `kivabe/kemne/kmne`, `bhalo/valo/vlo`). We deterministically generate
   the variants people really type — **train-split only** — so the model learns meaning,
   not surface form. See `data/sources.md`.
4. **Safety in-register.** Graceful refusals + crisis-support responses written in Banglish.

## Sourcing & licensing
- **100% original / synthetic** — authored pairs + safe slot-substitution + deterministic
  spelling variants. No scraped or third-party corpus. Full methodology in
  **`data/sources.md`**.
- **Dataset license:** CC-BY-4.0 (safe to release; nothing upstream restricts it).
- **PII:** all names/numbers/emails are fabricated placeholders. No real personal data.

## Reproduce
```bash
python data/build_dataset.py          # deterministic (seed=42) → all files below
python data/build_dataset.py --stats-only
```
Outputs: `banglish_instructions.jsonl` / `.csv`, `train.jsonl`, `heldout.jsonl`,
`sample.jsonl`. Pure stdlib, no dependencies, no network.

## Ethical considerations
- Advice pairs (health/finance/legal) are general and point to professionals/authorities
  where appropriate; the safety-critical rows (chest pain, self-harm, scams) escalate to
  real helplines (India: 108 ambulance, 1930 cyber, Tele-MANAS 14416).
- The set is a **seed to prove lift**, then scale — see the scaling recipe in
  `data/sources.md`.
