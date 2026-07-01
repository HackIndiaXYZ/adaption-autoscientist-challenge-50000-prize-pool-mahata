# Problem Statement — BanglaBridge

## Short version (paste into the HackIndia "Problem Statement" field)

BanglaBridge: a domain-adapted LLM for code-mixed, romanized Bengali ("Banglish") — the
way 100M+ people actually type Bengali online. Off-the-shelf models garble it. Using
Adaption's Adaptive Data + AutoScientist loop, we build an original, cleaned Banglish
instruction dataset and fine-tune a model that measurably beats the Language baseline on
Adaption's held-out test set. Weights + dataset released openly on Hugging Face and Kaggle
with a full model card and reproducible pipeline.

## Long version

**The gap.** Bengali is the 6th–7th most spoken language on Earth, but the way it lives
online is *romanized and code-mixed with English* ("kal ki plan? ami free achi"). Mainstream
LLMs are trained mostly on clean, native-script text, so they misread, mistranslate, and
fail to follow instructions in this register. The people most affected — everyday Bengali
speakers, students, small businesses — are exactly the ones underserved by "average use case"
models.

**The build.** An instruction-tuned model specialized for code-mixed Bengali, produced with
Adaption's platform end-to-end:
1. **Adaptive Data** — ingest, clean, and adapt an original Banglish instruction corpus
   (romanized + code-switched), with quality/eval passes on the platform.
2. **AutoScientist** — co-optimize the dataset and training recipe until the model converges
   on strong instruction-following in the target register.

**The proof.** A measurable percentage improvement over Adaption's Language baseline on their
held-out test set — the eligibility gate — documented in `eval/`.

**The release.** Open weights + adapted dataset on Hugging Face *and* Kaggle, model card,
dataset card, reproducible pipeline, and a live demo. Posted on LinkedIn + X tagging
@adaption_ai.

> Note: the *exact* target register/task (pure Banglish vs. transliteration vs. Bengali-English
> QA) is finalized after inspecting Adaption's actual Language baseline + test-set format at
> sign-up. This statement holds regardless; only the fine-grained task focus is tuned to it.
