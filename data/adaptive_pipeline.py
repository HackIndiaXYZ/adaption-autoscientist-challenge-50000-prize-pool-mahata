"""
BanglaBridge — Adaptive Data pipeline (real Adaption SDK)
=========================================================
Verified against Adaption's quickstart:
  https://adaptionlabs.ai/blog/adaptive-data-api-and-python-sdk
  https://docs.adaptionlabs.ai/

Flow:  upload -> wait ingest -> ESTIMATE (free) -> [confirm] -> run (spends
       credits) -> wait -> read improvement -> download adapted set.

Setup:
  pip install adaption
  # key is read from ../.env (ADAPTION_API_KEY=...), which is gitignored.

Safety:
  * `--estimate-only`  previews credits and spends NOTHING (default-safe habit).
  * the real adaptation only runs when you pass `--yes`.
  * start with a small `--rows` to gauge cost, then scale to all rows.

Examples:
  python data/adaptive_pipeline.py --rows 20 --estimate-only   # free cost check
  python data/adaptive_pipeline.py --rows 20 --yes             # adapt 20 rows
  python data/adaptive_pipeline.py --yes                       # adapt ALL rows
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent
ROOT = DATA.parent


def load_env() -> None:
    """Minimal stdlib .env loader — so the key is never hardcoded or committed."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapt the BanglaBridge dataset via Adaption.")
    ap.add_argument("--input", default=str(DATA / "banglish_instructions.csv"),
                    help="CSV/JSONL with columns original_prompt,response")
    ap.add_argument("--rows", type=int, default=0, help="max_rows to adapt (0 = all)")
    ap.add_argument("--estimate-only", action="store_true",
                    help="preview credit cost and exit — spends nothing")
    ap.add_argument("--yes", action="store_true",
                    help="actually run the adaptation (SPENDS credits)")
    ap.add_argument("--resume", default="",
                    help="dataset_id of an already-launched job: skip upload/run, "
                         "just wait for it and download (spends nothing new)")
    args = ap.parse_args()

    load_env()
    if not os.environ.get("ADAPTION_API_KEY"):
        raise SystemExit("ADAPTION_API_KEY not set. Put it in ../.env or the environment.")

    from adaption import Adaption  # imported after key check for a cleaner error

    client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
    mapping = {"prompt": "original_prompt"}
    job_spec: dict = {} if args.rows <= 0 else {"max_rows": args.rows}
    scope = "ALL" if args.rows <= 0 else str(args.rows)

    if args.resume:
        dataset_id = args.resume
        print(f"Resuming already-launched job {dataset_id} ...")
    else:
        # 1. UPLOAD -----------------------------------------------------------
        print(f"Uploading {args.input} ...")
        upload = client.datasets.upload_file(args.input)
        dataset_id = upload.dataset_id
        print(f"  dataset_id = {dataset_id}")

        # 2. WAIT FOR INGEST --------------------------------------------------
        while True:
            status = client.datasets.get_status(dataset_id)
            if getattr(status, "row_count", None) is not None:
                break
            time.sleep(2)
        print(f"  ingested {status.row_count} rows")

        # 3. ESTIMATE (free) --------------------------------------------------
        est = client.datasets.run(dataset_id, column_mapping=mapping,
                                  job_specification=job_spec, estimate=True)
        print(f"\nESTIMATE for {scope} rows: "
              f"~{getattr(est, 'estimated_minutes', '?')} min, "
              f"~{getattr(est, 'estimated_credits_consumed', '?')} credits.")

        if args.estimate_only or not args.yes:
            print("\nNothing spent. Re-run with --yes to actually adapt "
                  "(add --rows N to limit scope).")
            return

        # 4. ADAPT (SPENDS CREDITS) --------------------------------------------
        job = client.datasets.run(dataset_id, column_mapping=mapping,
                                  job_specification=job_spec)
        print(f"\nLaunched — ~{job.estimated_minutes} min, "
              f"{job.estimated_credits_consumed} credits. Waiting...")

    # WAIT — tolerate transient network drops (wifi blips kill getaddrinfo)
    deadline = time.time() + 3600
    while True:
        try:
            client.datasets.wait_for_completion(dataset_id, timeout=3600)
            break
        except Exception as e:  # noqa: BLE001 — SDK raises APIConnectionError etc.
            if time.time() > deadline:
                raise
            print(f"  connection hiccup ({type(e).__name__}), retrying in 30s ...")
            time.sleep(30)

    # 5. READ IMPROVEMENT -----------------------------------------------------
    ds = client.datasets.get(dataset_id)
    summary = getattr(ds, "evaluation_summary", None)
    if summary:
        print(f"Quality: {summary.grade_before} -> {summary.grade_after}")
        if getattr(summary, "improvement_percent", None):
            print(f"Improvement: {summary.improvement_percent:.0f}%")

    # 6. EXPORT ---------------------------------------------------------------
    # download() returns the adapted dataset CONTENT (CSV text), not a URL.
    adapted = client.datasets.download(dataset_id)
    if isinstance(adapted, bytes):
        adapted = adapted.decode("utf-8")
    (DATA / "full").mkdir(exist_ok=True)
    out = DATA / "full" / "banglish_instructions.adapted.csv"
    out.write_text(adapted, encoding="utf-8")
    (DATA / "adapted_result.txt").write_text(
        f"dataset_id={dataset_id}\n", encoding="utf-8")
    print(f"\nSaved adapted dataset -> {out}  ({len(adapted):,} chars)")
    print("Run scripts/make_finetune.py to build the slim (no-embeddings) fine-tune file.")
    # NEXT: release this adapted set (HF + Kaggle), then feed to AutoScientist
    # training and evaluate vs the Language baseline on the held-out test set.


if __name__ == "__main__":
    main()
