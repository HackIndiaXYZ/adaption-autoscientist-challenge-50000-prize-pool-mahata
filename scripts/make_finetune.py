#!/usr/bin/env python3
"""
Build the slim fine-tuning file from Adaption's adapted export.

Input : data/full/banglish_instructions.adapted.csv
        (cols: original_prompt, response, enhanced_prompt, enhanced_completion, ...)
Output: data/banglish_finetune.adapted.jsonl
        {instruction, output, orig_instruction, orig_output}  — NO embeddings.

We train on the *enhanced* pair (Adaption's quality-lifted, native-leaning Bengali)
and keep the originals for reference/ablation. Pure stdlib.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SRC = DATA / "full" / "banglish_instructions.adapted.csv"
OUT = DATA / "banglish_finetune.adapted.jsonl"


def has_bengali(s: str) -> bool:
    return any("ঀ" <= c <= "৿" for c in s)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Run data/adaptive_pipeline.py --yes first.")

    rows = list(csv.DictReader(io.StringIO(SRC.read_text(encoding="utf-8"))))
    slim = []
    for r in rows:
        instr = (r.get("enhanced_prompt") or r.get("original_prompt") or "").strip()
        out = (r.get("enhanced_completion") or r.get("response") or "").strip()
        if not instr or not out:
            continue
        slim.append({
            "instruction": instr,
            "output": out,
            "orig_instruction": (r.get("original_prompt") or "").strip(),
            "orig_output": (r.get("response") or "").strip(),
        })

    with OUT.open("w", encoding="utf-8") as f:
        for s in slim:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    orig_native = sum(1 for s in slim if has_bengali(s["orig_instruction"]))
    enh_native = sum(1 for s in slim if has_bengali(s["instruction"]))
    print(f"Wrote {OUT}  ({len(slim)} rows)")
    print(f"Native-script prompts: original {orig_native} -> enhanced {enh_native}")


if __name__ == "__main__":
    main()
