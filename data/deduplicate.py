"""Conservative exact matching after Unicode and whitespace normalization."""

import hashlib
import re
import unicodedata


def problem_sha256(problem: str) -> str:
    text = " ".join(unicodedata.normalize("NFC", problem).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_benchmark_source(source: str) -> bool:
    name = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")
    return bool(re.search(r"(?:^|-)(?:gsm-?8k|math-?500|hendrycks-math)(?:-|$)", name)) or name in {
        "math", "math-train", "math-test", "math-validation",
    }
