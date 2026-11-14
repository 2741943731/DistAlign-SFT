"""Grade saved math completions with a strict, numeric pilot scorer."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re

from data.validate import Jsonl, text_field


BOXED = re.compile(r"\\boxed\b")
DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")
SLASH = re.compile(r"([+-]?[0-9]+)\s*/\s*([+-]?[0-9]+)")
LATEX = re.compile(r"([+-]?)\s*\\frac\s*\{\s*([+-]?[0-9]+)\s*\}\s*\{\s*([+-]?[0-9]+)\s*\}")


def extract_final_answer(completion: str) -> str | None:
    """Read the final boxed answer; a truncated final box is unparsed."""
    boxes = list(BOXED.finditer(completion))
    if not boxes:
        return None
    tail = completion[boxes[-1].end():].lstrip()
    if not tail.startswith("{"):
        return None

    depth = 1
    for index, char in enumerate(tail[1:], 1):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return tail[1:index].strip() or None
    return None


def parse_number(answer: str) -> Fraction | None:
    """Parse only integers, finite decimal strings, and simple fractions."""
    text = answer.strip().replace("−", "-")
    try:
        if DECIMAL.fullmatch(text):
            return Fraction(text)
        if match := SLASH.fullmatch(text):
            numerator, denominator = match.groups()
            return Fraction(int(numerator), int(denominator))
        if match := LATEX.fullmatch(text):
            sign, numerator, denominator = match.groups()
            return (-1 if sign == "-" else 1) * Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError):
        pass
    return None


def grade(completion: str, answer: str) -> dict:
    prediction = extract_final_answer(completion)
    if prediction is None:
        status = "unparsed"
    else:
        predicted = parse_number(prediction)
        expected = parse_number(answer)
        if predicted is None or expected is None:
            status = "unsupported"
        else:
            status = "correct" if predicted == expected else "incorrect"
    return {"prediction": prediction, "status": status, "correct": status == "correct"}


def grade_file(predictions: Path, output: Path) -> dict:
    results = []
    seen_ids = set()
    for row, where in Jsonl(predictions):
        identifier = text_field(row, "id", where)
        answer = text_field(row, "answer", where)
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise ValueError(f"{where}: 'completion' must be a string")
        if identifier in seen_ids:
            raise ValueError(f"{where}: duplicate id {identifier!r}")
        seen_ids.add(identifier)
        results.append({"id": identifier, "completion": completion, "answer": answer, **grade(completion, answer)})
    if not results:
        raise ValueError("at least one prediction is required")

    with output.open("x", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    counts = Counter(result["status"] for result in results)
    return {"total": len(results), "accuracy": counts["correct"] / len(results), "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with id, completion, and answer")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file for per-question results")
    args = parser.parse_args()
    try:
        summary = grade_file(args.predictions, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
