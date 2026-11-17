"""Build the pinned NuminaMath-CoT split for the single-GPU SFT pilot."""

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

from .deduplicate import problem_sha256
from .prepare import prepare
from .validate import Jsonl, teacher_fields, text_field


NUMINA_REPO = "project-numina/NuminaMath-CoT"
NUMINA_REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
NUMINA_SHARD = "data/train-00000-of-00005.parquet"
NUMINA_SHARD_SHA256 = "25c0fa2ca96dd8078470ad1bbc439d0d211fd8b7d22e04abf7f61e9be9bf9ec3"
NUMINA_COLUMNS = ("problem", "solution", "source")
DATASETS_VERSION = "4.4.1"
CANDIDATE_LIMIT = 16_384
TRAIN_ROWS = 10_240
DEV_ROWS = 256
SEED = 42
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_PROBLEMS_SHA256 = "3084c8bf2e2a0e3737de89f6340802cd95f8950d8ef4daf2d727ca41ed1d7d6e"


def check_holdout(path: Path) -> None:
    reader = Jsonl(path)
    problems = set()
    for row, where in reader:
        if set(row) != {"problem"}:
            raise ValueError(f"{where}: expected a question-only MATH-500 row")
        problems.add(problem_sha256(text_field(row, "problem", where)))
    if reader.rows != 500 or len(problems) != 500 or reader.provenance()["sha256"] != MATH500_PROBLEMS_SHA256:
        raise ValueError(f"{path}: does not match the pinned 500-question holdout")


def check_local_shard(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != NUMINA_SHARD_SHA256:
        raise ValueError(f"{path}: SHA-256 mismatch; expected {NUMINA_SHARD_SHA256}, got {actual}")
    return actual


def stream_teacher(local_shard: Path | None = None) -> Iterable[dict]:
    try:
        import datasets
    except ImportError as exc:
        raise ValueError(f"install datasets=={DATASETS_VERSION} in the data environment first") from exc
    if datasets.__version__ != DATASETS_VERSION:
        raise ValueError(f"expected datasets=={DATASETS_VERSION}; found {datasets.__version__}")

    url = (str(local_shard.resolve()) if local_shard is not None else
           f"https://huggingface.co/datasets/{NUMINA_REPO}/resolve/{NUMINA_REVISION}/{NUMINA_SHARD}")
    yield from datasets.load_dataset(
        "parquet", data_files={"train": url}, split="train", streaming=True, columns=list(NUMINA_COLUMNS),
    )


def build_split(
    rows: Iterable[dict], held_out: Path, output_dir: Path, *,
    candidate_limit: int, train_rows: int, dev_rows: int,
) -> dict:
    if min(train_rows, dev_rows) < 1 or candidate_limit < train_rows + dev_rows:
        raise ValueError("candidate limit must cover both requested splits")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True)
    teacher_path = output_dir / "teacher.jsonl"
    with teacher_path.open("x", encoding="utf-8", newline="\n") as stream:
        for n, item in enumerate(islice(rows, candidate_limit), 1):
            row = teacher_fields({key: item.get(key) for key in NUMINA_COLUMNS}, f"Numina shard row {n}")
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if n % 4_096 == 0:
                print(f"Read {n} Numina rows", file=sys.stderr, flush=True)

    total = train_rows + dev_rows
    audit = prepare(
        teacher_path, [held_out], output_dir / "split", seed=SEED,
        # Aim at an interior point of the range that floors to dev_rows.
        validation_fraction=(dev_rows + 0.5) / total, limit=total,
    )
    actual = {name: split["rows"] for name, split in audit["splits"].items()}
    if actual != {"train": train_rows, "validation": dev_rows}:
        raise ValueError(f"insufficient eligible Numina rows: found {actual}; try a larger candidate limit in a new run")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", required=True, type=Path, help="Pinned, question-only MATH-500 JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh directory for this data build")
    parser.add_argument("--candidate-limit", type=int, default=CANDIDATE_LIMIT, help="Maximum rows to read from the first shard")
    parser.add_argument("--local-shard", type=Path, help="Offline copy of the pinned first Parquet shard; SHA-256 must match")
    args = parser.parse_args()
    try:
        check_holdout(args.held_out)
        local_shard_sha = check_local_shard(args.local_shard) if args.local_shard is not None else None
        audit = build_split(
            stream_teacher(args.local_shard), args.held_out, args.output_dir,
            candidate_limit=args.candidate_limit, train_rows=TRAIN_ROWS, dev_rows=DEV_ROWS,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    manifest = {
        "protocol": "numina-sft-single-gpu-pilot-v1",
        "source": {"repo": NUMINA_REPO, "revision": NUMINA_REVISION, "shard": NUMINA_SHARD,
                   "columns": NUMINA_COLUMNS, "datasets_version": DATASETS_VERSION,
                   "candidate_limit": args.candidate_limit, "raw_rows": audit["input"]["rows"],
                   "local_shard": str(args.local_shard.resolve()) if args.local_shard else None,
                   "local_shard_sha256": local_shard_sha},
        "held_out": {"repo": "HuggingFaceH4/MATH-500", "revision": MATH500_REVISION,
                     "problems_sha256": MATH500_PROBLEMS_SHA256, "role": "exact problem exclusion only"},
        "selection_seed": SEED,
        "excluded": audit["excluded"],
        "splits": audit["splits"],
        "source_counts": {
            name: dict(sorted(Counter(row["source"] for row, _ in Jsonl(args.output_dir / "split" / split["file"])).items()))
            for name, split in audit["splits"].items()
        },
        "validation_status": "teacher solutions not independently verified; audit before using for model selection",
        "sequence_length_filter_applied": False,
    }
    raw_manifest = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (args.output_dir / "manifest.json").write_text(raw_manifest, encoding="utf-8")
    print(raw_manifest, end="")
    print(f"Data directory: {args.output_dir}")


if __name__ == "__main__":
    main()
