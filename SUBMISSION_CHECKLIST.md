# Submission Checklist — Adaption AutoScientist × HackIndia

Every item below is required for **prize eligibility**. Print this and tick as you go.

## Admin (do first — unblocks everything)
- [ ] Joined the mandatory WhatsApp channel
- [ ] Joined Adaption Discord (`#autoscient-challenge` channel)
- [ ] Signed up at **adaptionlabs.ai** → 1,000 credits activated
- [ ] Confirmed HackIndia registration (Team **MAHATA**) is active
- [ ] Problem statement added to HackIndia event page (see `PROBLEM_STATEMENT.md`)
- [ ] This repo pushed to the HackIndia submission GitHub URL

## Recon (before building)
- [ ] Picked category: **Language** (Part 1)
- [ ] Inspected Adaption's **baseline model** for Language
- [ ] Inspected the **held-out test-set format** + scoring metric
- [ ] Finalized exact target task/register based on that test set

## Build
- [ ] Original Banglish dataset ingested + adapted in **Adaptive Data**
- [ ] Train / held-out split done, no leakage into eval
- [ ] Ran the **AutoScientist** data↔recipe co-optimization loop (logged in `training/`)
- [ ] Model converged; adapters/weights saved

## The gate
- [ ] **Measurable % improvement over baseline** on Adaption's held-out test set
- [ ] Results table filled in `MODEL_CARD.md` + `eval/`

## Open release (both platforms required)
- [ ] Weights on **Hugging Face** + model card
- [ ] Dataset on **Hugging Face** + dataset card
- [ ] Weights on **Kaggle** + model card
- [ ] Dataset on **Kaggle** + dataset card
- [ ] Reproducible pipeline documented

## Amplify
- [ ] LinkedIn post (tag **Adaption**)
- [ ] X post (tag **@adaption_ai**)
- [ ] (Bonus) live demo shipped + linked

## Submit
- [ ] **Part 1 submission form** filed before **July 5**
- [ ] All links (HF, Kaggle, repo, demo, posts) included in submission

## Deadlines
- Part 1 (Language): closes **July 5** · winners **July 13**
- Part 2 (if you also enter): **July 6 → Aug 3** · winners **Aug 10**
