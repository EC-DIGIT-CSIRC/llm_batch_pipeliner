#!/usr/bin/env python3
"""Create a deterministic stratified benchmark subset from an existing batch.

The source batch must already have:
- input/*.eml
- evaluation/category-map.json
- prompt.txt
- schema.py
- config.toml

Selection is based on the class derived from the filename prefix before `__`
and the category map. This script is designed for the SpamAssassin-style batch
layout used in this repo.

Example:
    scripts/make_spam_benchmark_subset.py \
        batches/batch_009_spam_benchmark_3shard_dashboard_repeat \
        batches/batch_030_qwen3_6_tp_300 \
        --ham 180 --spam 120 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_batch", type=Path, help="Existing benchmark batch directory")
    parser.add_argument("target_batch", type=Path, help="New subset batch directory to create")
    parser.add_argument("--ham", type=int, default=180, help="Number of ham samples (default: 180)")
    parser.add_argument("--spam", type=int, default=120, help="Number of spam samples (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_batch
    target = args.target_batch

    if target.exists():
        raise SystemExit(f"Target already exists: {target}")

    category_map = json.loads((source / "evaluation" / "category-map.json").read_text(encoding="utf-8"))
    # The existing SpamAssassin benchmark batch stores the corpus with bare
    # filenames like `easy_ham__00027...` (no `.eml` suffix), so we intentionally
    # select all non-directory entries here instead of globbing `*.eml`.
    inputs = sorted(p for p in (source / "input").iterdir() if p.is_file())

    ham_files: list[Path] = []
    spam_files: list[Path] = []
    for path in inputs:
        prefix = path.name.split("__", 1)[0]
        label = category_map.get(prefix)
        if label == "ham":
            ham_files.append(path)
        elif label == "spam":
            spam_files.append(path)

    rnd = random.Random(args.seed)
    rnd.shuffle(ham_files)
    rnd.shuffle(spam_files)

    if len(ham_files) < args.ham:
        raise SystemExit(f"Not enough ham files: need {args.ham}, found {len(ham_files)}")
    if len(spam_files) < args.spam:
        raise SystemExit(f"Not enough spam files: need {args.spam}, found {len(spam_files)}")

    selected = sorted(ham_files[: args.ham] + spam_files[: args.spam], key=lambda p: p.name)

    (target / "input").mkdir(parents=True)
    (target / "evaluation").mkdir(parents=True)
    (target / "logs").mkdir(parents=True)

    for name in ("prompt.txt", "schema.py", "config.toml"):
        shutil.copy2(source / name, target / name)
    shutil.copy2(source / "evaluation" / "category-map.json", target / "evaluation" / "category-map.json")
    for src in selected:
        shutil.copy2(src, target / "input" / src.name)

    manifest = {
        "source_batch": str(source),
        "seed": args.seed,
        "ham_count": args.ham,
        "spam_count": args.spam,
        "total": len(selected),
        "selected": [p.name for p in selected],
    }
    (target / "subset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
