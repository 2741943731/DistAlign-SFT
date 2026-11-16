"""Conservative whole-answer comparison for the paper-style benchmark runner."""

from fractions import Fraction
from functools import lru_cache
from importlib import metadata
import re

from .grade import extract_final_answer, parse_number


DEPENDENCY = "math-verify[antlr4_13_2]==0.9.0"
SCORER = "math-verify-0.9.0-whole-box-v1"
_VERSIONS = {
    "math-verify": "0.9.0", "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2", "sympy": "1.14.0", "mpmath": "1.3.0",
}
_DELIMITERS = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))
_PERCENT = re.compile(r"(.+?)\s*(?:\\?%)")
_SCALE = re.compile(r"\\(left|right)\s*(\\[{}|]|[()\[\]|])")
_UNCLEAR = re.compile(
    r"\$|\\(?:boxed|fbox|text|mbox|mathrm|textnormal|textbf|textit|approx|sim|simeq|ldots|cdots)\b|[≈≊≋≃≅]|~="
)


def _number(body: str) -> Fraction | None:
    match = _PERCENT.fullmatch(body.strip())
    if match:
        value = parse_number(match[1])
        return None if value is None else value / 100
    return parse_number(body)


def _body(text: str) -> str | None:
    body = text.strip().replace("−", "-")
    for opening, closing in _DELIMITERS:
        if body.startswith(opening) and body.endswith(closing) and len(body) > len(opening) + len(closing):
            body = body[len(opening):-len(closing)].strip()
            break
    if not body or len(body) > 512 or _UNCLEAR.search(body) or body.count("=") > 1:
        return None
    # Drop paired delimiter-size commands, not their brackets. This retains
    # open/closed interval endpoints and leaves unmatched commands invalid.
    depth = 0
    for match in _SCALE.finditer(body):
        depth += 1 if match[1] == "left" else -1
        if depth < 0:
            return None
    if depth:
        return None
    body = _SCALE.sub(r"\2", body)
    if re.search(r"\\(?:left|right)\b", body):
        return None
    return body


@lru_cache(maxsize=1)
def _backend():
    try:
        for package, expected in _VERSIONS.items():
            actual = metadata.version(package)
            if actual != expected:
                raise ValueError(f"{package}=={expected} is required; found {actual}")
        import sympy as sp
        from sympy.core.relational import Relational
        from latex2sympy2_extended import latex2sympy
        from latex2sympy2_extended.latex2sympy2 import ConversionConfig
        from math_verify import verify
        from math_verify.errors import TimeoutException
        from math_verify.utils import timeout
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise ValueError(f"paper scoring requires {DEPENDENCY}") from exc

    options = ConversionConfig(
        interpret_as_mixed_fractions=False,
        interpret_simple_eq_as_assignment=False,
        interpret_contains_as_eq=False,
        lowercase_symbols=False,
    )

    @timeout(5)
    def parse(body, number):
        if number is not None:
            return sp.Rational(number.numerator, number.denominator)
        # A juxtaposed integer and LaTeX fraction has ambiguous semantics.
        # Percentages are only supported by the exact standalone path above:
        # the upstream equivalence comparator can otherwise elide the scale.
        if "%" in body or re.search(r"[0-9]\s*\\(?:[dtc]?frac)\b", body):
            return None
        value = latex2sympy(body, normalization_config=None, conversion_config=options)
        if not isinstance(value, sp.Basic) or value.has(sp.nan, sp.zoo):
            return None
        # Treat finite decimal atoms as exact, including inside LaTeX fractions.
        floats = {atom: sp.Rational(str(atom)) for atom in value.atoms(sp.Float)}
        if floats:
            value = value.xreplace(floats)
        if value.has(sp.nan, sp.zoo) or any(
            atom.base.is_zero is True and atom.exp.is_negative is True
            for atom in value.atoms(sp.Pow)
        ):
            return None
        return value

    def family(value):
        if isinstance(value, Relational):
            if not value.free_symbols:
                return None
            return "relation"
        if isinstance(value, sp.Expr):
            return "scalar"
        if isinstance(value, sp.Interval):
            return "interval"
        if isinstance(value, (sp.FiniteSet, sp.Tuple)) and all(isinstance(x, sp.Expr) for x in value):
            return "set" if isinstance(value, sp.FiniteSet) else "tuple"
        return None

    def compare_parsed(pred, gold, pred_number, gold_number):
        try:
            prediction, answer = parse(pred, pred_number), parse(gold, gold_number)
            prediction_family, answer_family = family(prediction), family(answer)
            if prediction_family is None or answer_family is None:
                return None
            if prediction_family != answer_family:
                return False
            if answer_family == "relation" and answer.free_symbols != prediction.free_symbols:
                return False
            if answer_family == "interval" and (
                (answer.left_open, answer.right_open) != (prediction.left_open, prediction.right_open)
            ):
                return False
            return verify(answer, prediction, strict=True, allow_set_relation_comp=False, timeout_seconds=5, raise_on_error=True)
        except (Exception, TimeoutException):
            # The upstream parser reports malformed LaTeX with bare Exception.
            return None

    return compare_parsed


def check_dependencies() -> None:
    """Validate the pinned optional backend before starting an expensive run."""
    _backend()


def compare(completion: str, answer: str) -> dict:
    prediction = extract_final_answer(completion)
    if prediction is None:
        return {"prediction": None, "status": "unparsed", "correct": False}
    pred, gold = _body(prediction), _body(answer)
    if pred is None or gold is None:
        matches = None
    else:
        pred_number, gold_number = _number(pred), _number(gold)
        if pred_number is not None and gold_number is not None:
            matches = pred_number == gold_number
        else:
            matches = _backend()(pred, gold, pred_number, gold_number)
    status = "unsupported" if matches is None else "correct" if matches else "incorrect"
    return {"prediction": prediction, "status": status, "correct": status == "correct"}
