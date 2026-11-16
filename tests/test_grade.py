from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest

from src.grade import extract_final_answer, grade, grade_file


class GradeTests(unittest.TestCase):
    def test_nested_box_and_equivalent_numbers(self):
        self.assertTrue(grade(r"Thus \boxed{\frac{1}{2}}.", "0.5")["correct"])
        self.assertTrue(grade(r"Thus \boxed{-\frac{2}{4}}.", "-.5")["correct"])

    def test_does_not_credit_intermediate_boxes_or_unmarked_numbers(self):
        self.assertEqual(grade(r"Try \boxed{3}. Finally \boxed{4}.", "3")["status"], "incorrect")
        self.assertIsNone(extract_final_answer(r"Try \boxed{3}. Finally \boxed{\frac{1}{"))
        self.assertIsNone(extract_final_answer("I think it is 3."))

    def test_symbolic_and_undefined_answers_are_unsupported(self):
        self.assertEqual(grade(r"\boxed{\sqrt{8}}", r"2\sqrt{2}")["status"], "unsupported")
        self.assertEqual(grade(r"\boxed{1/0}", "1")["status"], "unsupported")

    def test_per_question_report_and_denominator(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "grading.jsonl"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "graded.jsonl"
            summary = grade_file(fixture, output)
            results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["accuracy"], 0.4)
        self.assertEqual(summary["counts"], {"correct": 2, "incorrect": 1, "unparsed": 1, "unsupported": 1})
        self.assertEqual(results[2]["prediction"], "4")

    @unittest.skipUnless(find_spec("lark") and find_spec("sympy"), "symbolic-eval is not installed")
    def test_symbolic_grades_entire_box_without_relaxing_extraction(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "grading-symbolic.jsonl"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "graded.jsonl"
            summary = grade_file(fixture, output, scorer="symbolic")
            results = {r["id"]: r["status"] for r in map(json.loads, output.read_text().splitlines())}
        self.assertEqual(results, {
            "radical": "correct", "pi": "correct", "scalar": "correct",
            "percent": "correct", "percent-needs-conversion": "incorrect",
            "different-radical": "incorrect", "different-exact-value": "incorrect",
            "malformed": "unsupported", "equation": "unsupported", "undefined": "unsupported",
            "ambiguous": "unsupported", "unboxed": "unparsed", "truncated": "unparsed",
        })
        self.assertEqual(summary["accuracy"], 4 / 13)
        self.assertEqual(summary["scorer"], "symbolic-v1")
