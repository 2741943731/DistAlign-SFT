"""Prepare an auditable, deterministic Phase 0 train/validation split."""

import argparse
import hashlib
import json
from pathlib import Path

from .deduplicate import is_benchmark_source, problem_sha256
from .validate import Jsonl, teacher_fields, text_field


def prepare(input_path: Path, held_out_paths: list[Path], output_dir: Path, *,
            seed: int = 42, validation_fraction: float = 0.1, limit: int | None = None) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1 (exclusive)")
    if limit is not None and limit < 2:
        raise ValueError("--limit must be at least 2 so both splits can be nonempty")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists: {output_dir}")

    held_out = set()
    held_out_files = []
    for path in held_out_paths:
        reader = Jsonl(path)
        for item, where in reader:
            held_out.add(problem_sha256(text_field(item, "problem", where)))
        held_out_files.append(reader.provenance())
    if not held_out:
        raise ValueError("at least one held-out problem is required")

    teacher = Jsonl(input_path)
    skipped = {"benchmark_source": 0, "held_out_exact": 0, "teacher_duplicate": 0, "above_limit": 0}
    eligible = {}
    for item, where in teacher:
        row = teacher_fields(item, where)
        if is_benchmark_source(row["source"]):
            skipped["benchmark_source"] += 1
            continue
        fingerprint = problem_sha256(row["problem"])
        if fingerprint in held_out:
            skipped["held_out_exact"] += 1
            continue
        if fingerprint in eligible:
            skipped["teacher_duplicate"] += 1
            continue
        row.setdefault("id", fingerprint)
        row["problem_sha256"] = fingerprint
        eligible[fingerprint] = row

    fingerprints = sorted(eligible, key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).digest(), value))
    if limit is not None:
        skipped["above_limit"] = max(0, len(fingerprints) - limit)
        fingerprints = fingerprints[:limit]
    if len(fingerprints) < 2:
        raise ValueError("at least two eligible, distinct teacher problems are required")
    n_validation = max(1, min(len(fingerprints) - 1, int(len(fingerprints) * validation_fraction)))
    splits = {"validation": fingerprints[:n_validation], "train": fingerprints[n_validation:]}
    audit = {
        "schema_version": 1,
        "input": teacher.provenance(),
        "held_out_files": held_out_files,
        "unique_held_out_problems": len(held_out),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "limit": limit,
        "excluded": skipped,
        "splits": {},
        "matching": "Exact SHA-256 after Unicode NFC and whitespace normalization; case preserved.",
        "duplicate_policy": "First eligible teacher row in input order for a normalized question.",
        "limitations": ["No near-duplicate, paraphrase, or translation matching."],
    }

    output_dir.mkdir(parents=True)
    for name, split in splits.items():
        path = output_dir / f"{name}.jsonl"
        checksum = hashlib.sha256()
        with path.open("xb") as stream:
            for fingerprint in split:
                line = (json.dumps(eligible[fingerprint], ensure_ascii=False, sort_keys=True) + "\n").encode()
                stream.write(line)
                checksum.update(line)
        audit["splits"][name] = {"file": path.name, "rows": len(split), "sha256": checksum.hexdigest()}
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Teacher JSONL with problem, solution, source, optional id.")
    parser.add_argument("--held-out", type=Path, action="append", required=True,
                        help="Benchmark JSONL with a problem field; repeat for each benchmark.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for train, validation, and audit files.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for stable problem ordering (default: 42).")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction (default: 0.1).")
    parser.add_argument("--limit", type=int, help="Cap the number of eligible unique problems before splitting.")
    args = parser.parse_args()
    try:
        audit = prepare(args.input, args.held_out, args.output_dir,
                        seed=args.seed, validation_fraction=args.validation_fraction, limit=args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {audit['splits']['train']['rows']} train and {audit['splits']['validation']['rows']} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
