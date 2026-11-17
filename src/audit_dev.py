"""Prepare teacher-answer candidates for manual review, without scoring models."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from data.validate import Jsonl, text_field
from .grade import extract_final_answer, parse_number


def prepare_audit(input_path: Path, output_dir: Path, *, sample_size: int = 12,
                  seed: int = 42, expected_sha256: str | None = None) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists: {output_dir}")
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    reader = Jsonl(input_path)
    records, ids = [], set()
    for item, where in reader:
        fields = {key: text_field(item, key, where) for key in ("id", "problem", "solution", "source")}
        if fields["id"] in ids:
            raise ValueError(f"{where}: duplicate id")
        ids.add(fields["id"])
        answer = extract_final_answer(fields["solution"])
        if answer is None:
            status = "no_complete_box"
        elif parse_number(answer) is None:
            status = "boxed_other"
        else:
            status = "boxed_numeric"
        records.append({**fields, "answer_candidate": answer, "extraction_status": status,
                        "manual_review_status": "not_reviewed"})
    if not records:
        raise ValueError("the validation data is empty")
    provenance = reader.provenance()
    if expected_sha256 is not None and provenance["sha256"] != expected_sha256:
        raise ValueError(f"validation SHA-256 mismatch: {provenance['sha256']}")

    # Selection depends only on fixed IDs and seed, never on model predictions.
    review = sorted(records, key=lambda r: (hashlib.sha256(f"{seed}:{r['id']}".encode()).digest(), r["id"]))[:sample_size]
    counts = Counter(row["extraction_status"] for row in records)
    report = {
        "input": provenance,
        "counts": {key: counts[key] for key in ("boxed_numeric", "boxed_other", "no_complete_box")},
        "sample_size": len(review), "sample_seed": seed,
        "sample_selection": "SHA-256 of seed:id sorted ascending; independent of model outputs",
        "interpretation": "Extracted candidates are not independently verified. Manual review is required.",
    }
    output_dir.mkdir(parents=True)
    for name, values in (("all-candidates.jsonl", records), ("review-sample.jsonl", review)):
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "review-sample.txt").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(review, 1):
            stream.write(f"题目 {index} | ID: {row['id']} | 来源: {row['source']}\n")
            stream.write(f"题面:\n{row['problem']}\n\n教师解答（未经核验）:\n{row['solution']}\n\n")
            stream.write(f"提取的候选答案: {row['answer_candidate']}\n")
            stream.write(f"提取状态: {row['extraction_status']}\n人工核验: 未核验\n\n")
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True, help="Teacher validation JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for review files")
    parser.add_argument("--expected-sha256", help="Expected SHA-256 of the validation JSONL")
    parser.add_argument("--sample-size", type=int, default=12, help="Number of rows to select for manual review")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_audit(args.validation, args.output_dir, sample_size=args.sample_size,
                               seed=args.seed, expected_sha256=args.expected_sha256)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Manual review sample: {args.output_dir / 'review-sample.txt'}")


if __name__ == "__main__":
    main()
