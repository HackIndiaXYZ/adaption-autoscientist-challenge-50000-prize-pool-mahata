#!/usr/bin/env python3
"""
Publish BanglaBridge-Instruct to the Hugging Face Hub — model adapter + dataset.

One command, two repos:
  {user}/BanglaBridge-Instruct     (LoRA adapter + model card)
  {user}/banglabridge-instructions (dataset + dataset card)

Auth (either works):
  - set HF_TOKEN in the environment, or
  - pass --token hf_xxx
Get a WRITE token at: huggingface.co/settings/tokens  (role: Write)

Usage:
  python training/push_to_hf.py --user <your-hf-username>
  python training/push_to_hf.py --user <your-hf-username> --token hf_xxx --private
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "training" / "banglabridge-adapter"
MODEL_CARD = ROOT / "MODEL_CARD.md"
DATASET_CARD = ROOT / "DATASET_CARD.md"
DATASET_FILE = ROOT / "data" / "banglish_finetune.adapted.jsonl"

MODEL_REPO = "BanglaBridge-Instruct"
DATASET_REPO = "banglabridge-instructions"


def model_frontmatter(dataset_repo_id: str) -> str:
    return (
        "---\n"
        "base_model: togethercomputer/Meta-Llama-3.3-70B-Instruct-Reference\n"
        "library_name: peft\n"
        "license: llama3.3\n"
        "language:\n- bn\n- en\n"
        "tags:\n- lora\n- peft\n- bengali\n- banglish\n- code-mixed\n"
        "- instruction-tuning\n- adaption-autoscientist\n"
        f"datasets:\n- {dataset_repo_id}\n"
        "pipeline_tag: text-generation\n"
        "---\n\n"
    )


def dataset_frontmatter() -> str:
    return (
        "---\n"
        "license: cc-by-4.0\n"
        "language:\n- bn\n- en\n"
        "tags:\n- bengali\n- banglish\n- code-mixed\n- instruction-tuning\n"
        "- synthetic\n"
        "task_categories:\n- text-generation\n- question-answering\n"
        "pretty_name: BanglaBridge Instructions (Banglish)\n"
        "size_categories:\n- n<1K\n"
        "---\n\n"
    )


def build_readme(dest: Path, frontmatter: str, body_file: Path) -> None:
    body = body_file.read_text(encoding="utf-8")
    dest.write_text(frontmatter + body, encoding="utf-8")
    print(f"  wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face username or org")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--private", action="store_true",
                    help="create repos private (challenge requires PUBLIC before submit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the README files but do not upload")
    args = ap.parse_args()

    model_id = f"{args.user}/{MODEL_REPO}"
    dataset_id = f"{args.user}/{DATASET_REPO}"

    # Sanity checks -----------------------------------------------------------
    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        sys.exit(f"Missing adapter weights in {ADAPTER_DIR}")
    if not DATASET_FILE.exists():
        sys.exit(f"Missing dataset {DATASET_FILE}")

    # Assemble HF-ready cards -------------------------------------------------
    print("Assembling model card (README with PEFT frontmatter) ...")
    build_readme(ADAPTER_DIR / "README.md", model_frontmatter(dataset_id), MODEL_CARD)

    ds_staging = ROOT / "training" / "hf_dataset"
    ds_staging.mkdir(exist_ok=True)
    build_readme(ds_staging / "README.md", dataset_frontmatter(), DATASET_CARD)

    import shutil
    shutil.copy(DATASET_FILE, ds_staging / DATASET_FILE.name)

    if args.dry_run:
        print("\n[dry-run] Cards assembled. No upload. "
              "Re-run without --dry-run and with a token to publish.")
        return

    if not args.token:
        sys.exit("No token. Set HF_TOKEN or pass --token hf_xxx "
                 "(Write role, from huggingface.co/settings/tokens).")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    who = api.whoami()["name"]
    print(f"Authenticated as: {who}")

    # 1) MODEL adapter --------------------------------------------------------
    print(f"\nCreating + uploading model repo: {model_id}")
    api.create_repo(model_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ADAPTER_DIR),
        repo_id=model_id,
        repo_type="model",
        commit_message="BanglaBridge-Instruct: LoRA adapter for Llama 3.3 70B (65% win rate)",
    )
    print(f"  OK https://huggingface.co/{model_id}")

    # 2) DATASET --------------------------------------------------------------
    print(f"\nCreating + uploading dataset repo: {dataset_id}")
    api.create_repo(dataset_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ds_staging),
        repo_id=dataset_id,
        repo_type="dataset",
        commit_message="BanglaBridge instruction dataset (adapted, Banglish)",
    )
    print(f"  OK https://huggingface.co/datasets/{dataset_id}")

    print("\nDONE. Both repos live. If created --private, flip them PUBLIC in "
          "repo Settings before submitting (challenge requires open release).")


if __name__ == "__main__":
    main()
