from importlib.util import find_spec
import unittest

from src.paper_grade import check_dependencies, compare


class PaperGradeTests(unittest.TestCase):
    def test_exact_numeric_path_and_strict_box_extraction(self):
        self.assertEqual(compare(r"\boxed{0.500000000001}", "1/2")["status"], "incorrect")
        self.assertEqual(compare(r"\boxed{40.5\%}", "0.405")["status"], "correct")
        self.assertEqual(compare(r"\boxed{40.5\%}", "40.5")["status"], "incorrect")
        self.assertEqual(compare("Answer: 4.", "4")["status"], "unparsed")
        self.assertEqual(compare(r"\boxed{4}. Finally \boxed{\sqrt{", "4")["status"], "unparsed")

    @unittest.skipUnless(find_spec("math_verify"), "paper scorer dependencies are not installed")
    def test_symbolic_path_matches_whole_expressions_within_their_family(self):
        check_dependencies()
        cases = (
            (r"\sqrt{8}", r"2\sqrt{2}", "correct"),
            (r"\sqrt{3}", r"2\sqrt{2}", "incorrect"),
            (r"\left\{2,1\right\}", r"\{1,2\}", "correct"),
            (r"\{1,2,3\}", r"\{1,2\}", "incorrect"),
            (r"\{1,2\}", r"(1,2)", "incorrect"),
            (r"(1,2,3)", r"\{1,2,3\}", "incorrect"),
            (r"[1,2]", r"(1,2)", "incorrect"),
            (r"x=2", r"2", "incorrect"),
            (r"y=2", r"x=2", "incorrect"),
            (r"99=1", r"1", "unsupported"),
            (r"\sqrt{8}-", r"2\sqrt{2}", "unsupported"),
            (r"1\frac{1}{2}", r"1/2", "unsupported"),
            (r"\frac{1}{0}", r"1", "unsupported"),
        )
        for prediction, gold, expected in cases:
            with self.subTest(prediction=prediction, gold=gold):
                result = compare(r"\boxed{" + prediction + "}", gold)
                self.assertEqual(result, {
                    "prediction": prediction, "status": expected, "correct": expected == "correct",
                })
