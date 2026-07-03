# Model Card — BanglaBridge-Instruct

<!-- This doubles as the README on the Hugging Face model repo. Keep it in sync. -->

## Model summary
- **Model:** BanglaBridge-Instruct — a **LoRA adapter** for Llama 3.3 70B Instruct.
- **Base model:** `togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference`
  (Meta Llama 3.3 70B Instruct). Fine-tuned via Adaption AutoScientist; platform
  run `adaption_llama_3_3_70b_instru_banglish_helpful_pairs_959f9e3a`.
- **Adaptation method:** Adaption's two-stage pipeline — (1) **Adaptive Data**
  quality-lift of the raw instruction pairs, (2) **AutoScientist** self-learning
  fine-tuning loop (data ↔ recipe co-optimization) on the adapted set.
- **Language(s):** Bengali — code-mixed / romanized "Banglish", native script,
  and mixed code-switch — plus English.
- **License:** Llama 3.3 Community License (inherited from the base model).
  Dataset released separately under CC-BY-4.0.
- **Developed by:** Team MAHATA — Adaption AutoScientist Challenge × HackIndia
- **Repo:** https://github.com/HackIndiaXYZ/adaption-autoscientist-challenge-50000-prize-pool-mahata

## Why this model exists
100M+ Bengali speakers type in romanized "Banglish" ("kal ki plan? ami free
achi") — a register with **no standard orthography** that off-the-shelf models
routinely garble. BanglaBridge-Instruct is instruction-tuned to understand all
three real-world registers (romanized / native script / code-switch) and the
spelling chaos within them (`ache/ase/achhe`, `kivabe/kemne/kmne`).

## Results (the headline)

| Metric | Result |
| --- | --- |
| **Win rate vs. baseline** (Adaption held-out evaluation, Language category) | **65%** |

The adapted model's responses beat the baseline model's in 65% of head-to-head
judgments on Adaption's in-house held-out test set — a measurable improvement
over baseline, satisfying the challenge's eligibility gate.

## Intended use
Instruction-following, Q&A, translation, rewriting, and generation in
code-mixed / romanized / native Bengali — chat assistants, content tools, and
support bots serving Bengali speakers who type the way people actually type.

**Out of scope:** high-stakes medical/legal/financial advice; safety-critical
decisions.

## Training data
Original, hand-authored Banglish instruction dataset (0% scraped), processed
through Adaption's Adaptive Data pipeline and released openly alongside the
model. Sourcing, register distribution, augmentation design, and licensing are
documented in `DATASET_CARD.md` and `data/`.

Key dataset design choices:
- **3-register hedge:** romanized + native-script + mixed code-switch pairs.
- **Spelling-variation augmentation** (train split only): deterministic,
  meaning-preserving orthographic variants so the model learns the meaning,
  not the surface form.
- **21 task types** including safety/refusal behavior in-register.

## Training procedure
- **Platform:** Adaption AutoScientist (managed fine-tuning of Llama 3.3 70B Instruct)
- **Method:** LoRA (PEFT), bf16. `peft_type=LORA`, `task_type=CAUSAL_LM`.
- **Hyperparameters** (from `adapter_config.json` / `trainer_state.json`):
  - rank `r = 64`, `lora_alpha = 128`, `lora_dropout = 0.05`, `bias = none`
  - target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  - 4 epochs, 516 steps, per-device batch size 1
  - **train loss 1.52 → 0.38** (smooth convergence, no divergence)
- **Data pipeline:** `data/adaptive_pipeline.py` (upload → estimate → adapt →
  export; resumable, spend-gated)
- **Artifacts:** LoRA adapter (`adapter_model.safetensors`, ~3.3 GB), tokenizer,
  chat template, and `trainer_state.json` released in `training/`.

### How to use (load the adapter on the base model)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Llama-3.3-70B-Instruct"   # or the Together reference build
tok  = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "MAHATA/BanglaBridge-Instruct")  # this adapter
```

## Limitations & risks
- Register-specialized: tuned toward conversational Bengali registers; may not
  beat generic models on formal literary Bengali.
- Inherits Llama 3.3 base-model biases; informal registers may carry
  social-media-style biases.
- Evaluated via win-rate preference judgments — not a factual-accuracy
  benchmark; verify factual outputs independently.

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
