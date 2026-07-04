#!/usr/bin/env python3
"""
Publish BanglaBridge-Instruct to Kaggle — weights + dataset (both required
by the challenge, alongside the Hugging Face release).

Both artifacts go up as Kaggle DATASETS (simplest reliable API surface):
  kaggle.com/datasets/{user}/banglabridge-instruct-lora   (adapter weights)
  kaggle.com/datasets/{user}/banglabridge-instructions    (instruction data)

Auth: put kaggle.json (from kaggle.com/settings -> Create New Token) at
  C:\\Users\\<you>\\.kaggle\\kaggle.json
Usage:
  python training/push_to_kaggle.py --user <your-kaggle-username>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "training" / "banglabridge-adapter"
DS_STAGING = ROOT / "training" / "hf_dataset"  # README + jsonl, built by push_to_hf.py


def stage(dirname: str, title: str, slug_user: str, slug: str,
          files: list[Path], license_name: str) -> Path:
    d = ROOT / "training" / dirname
    d.mkdir(exist_ok=True)
    for f in files:
        shutil.copy(f, d / f.name)
    meta = {
        "title": title,
        "id": f"{slug_user}/{slug}",
        "licenses": [{"name": license_name}],
    }
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Kaggle username")
    args = ap.parse_args()

    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        sys.exit(f"Missing adapter weights in {ADAPTER_DIR}")
    if not (DS_STAGING / "banglish_finetune.adapted.jsonl").exists():
        sys.exit("Missing staged dataset — run push_to_hf.py --dry-run first.")

    # 1) weights (adapter + configs + card)
    weights_files = [p for p in ADAPTER_DIR.iterdir() if p.is_file()]
    wdir = stage("kaggle_weights", "BanglaBridge-Instruct (LoRA, Llama 3.3 70B)",
                 args.user, "banglabridge-instruct-lora", weights_files, "other")

    # 2) instruction dataset (jsonl + card)
    ddir = stage("kaggle_dataset", "BanglaBridge Instructions (Banglish)",
                 args.user, "banglabridge-instructions",
                 [DS_STAGING / "README.md", DS_STAGING / "banglish_finetune.adapted.jsonl"],
                 "CC-BY-4.0")

    kaggle = [sys.executable, "-m", "kaggle"]
    rc1 = run(kaggle + ["datasets", "create", "-p", str(wdir), "--dir-mode", "skip"])
    rc2 = run(kaggle + ["datasets", "create", "-p", str(ddir), "--dir-mode", "skip"])
    if rc1 or rc2:
        sys.exit("One or more Kaggle uploads failed — see output above. "
                 "(Already-exists errors mean it's live; use `kaggle datasets version` to update.)")

    print("\nDONE:")
    print(f"  https://www.kaggle.com/datasets/{args.user}/banglabridge-instruct-lora")
    print(f"  https://www.kaggle.com/datasets/{args.user}/banglabridge-instructions")


if __name__ == "__main__":
    main()
