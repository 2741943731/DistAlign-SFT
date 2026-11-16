"""Compare a bounded subset of exact real-valued math expressions."""

from fractions import Fraction

from lark import Lark, Transformer, v_args
from lark.exceptions import UnexpectedInput, VisitError
import sympy as sp


_PARSER = Lark(r"""
?start: sum
      | NUMBER ("%" | "\\%") -> percent
      | "-" NUMBER ("%" | "\\%") -> negative_percent
?sum: product | sum "+" product -> add | sum "-" product -> sub
?product: unary
        | product ("*" | "\\cdot" | "\\times") unary -> mul
        | product juxtaposed -> mul
?unary: power | "-" unary -> neg | "+" unary
?power: atom | atom "^" exponent -> power
?juxtaposed: named | named "^" exponent -> power
?atom: NUMBER -> number
     | named
     | ("\\frac" | "\\dfrac" | "\\tfrac") group group -> ratio
?named: ("\\pi" | "π") -> pi
      | "\\sqrt" group -> root
      | "(" sum ")"
?group: "{" sum "}"
?exponent: EXP | "{" EXP "}"
EXP: /-?[0-9]/
NUMBER: /(?:[0-9]{1,12}(?:\.[0-9]{1,12})?|\.[0-9]{1,12})/
%import common.WS
%ignore WS
""", parser="lalr")


@v_args(inline=True)
class _Exact(Transformer):
    def number(self, token):
        return sp.Rational(str(token))

    def percent(self, token):
        return self.number(token) / 100

    def negative_percent(self, token):
        return -self.percent(token)

    def pi(self):
        return sp.pi

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def neg(self, a):
        return -a

    def ratio(self, a, b):
        if b.is_zero is not False:
            raise ValueError("denominator is zero or unresolved")
        return a / b

    def root(self, a):
        if a.is_nonnegative is not True or not a.is_Rational:
            raise ValueError("only nonnegative rational radicands are supported")
        if max(int(a.p).bit_length(), int(a.q).bit_length()) > 40:
            raise ValueError("radicand is too large")
        return sp.sqrt(a)

    def power(self, a, exponent):
        n = int(exponent)
        if n <= 0 and a.is_zero is not False:
            raise ValueError("undefined or unresolved power")
        return a ** n


def _parse(value: str | Fraction):
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    value = value.strip().replace("−", "-")
    # Bound expression size and expansion from untrusted model output.
    if len(value) > 128 or value.count("^") > 2 or value.count(r"\sqrt") > 4:
        return None
    try:
        tree = _PARSER.parse(value)
        result = _Exact().transform(tree)
    except UnexpectedInput:
        return None
    except VisitError as exc:
        if isinstance(exc.orig_exc, ValueError):
            return None
        raise
    return result if result.is_real is True and result.is_finite is True else None


def equivalent(prediction: str | Fraction, answer: str | Fraction) -> bool | None:
    """Return None if either answer is outside this exact scalar profile."""
    pred, gold = _parse(prediction), _parse(answer)
    if pred is None or gold is None:
        return None
    delta = sp.simplify(pred - gold)
    if delta == 0:
        return True
    return False if delta.is_zero is False else None
