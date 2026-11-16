import json
from pathlib import Path
import tempfile
import unittest

from data.numina_pilot import build_split


class NuminaPilotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.held_out = self.root / "holdout.jsonl"
        self.held_out.write_text('{"problem":"held out"}\n', encoding="utf-8")

    def test_bounded_prefix_split_excludes_holdout_and_ignores_extra_columns(self):
        def rows():
            for problem in ("held   out", "A", "A", "B", "C", "D", "E"):
                yield {"problem": problem, "solution": "1", "source": "olympiads", "messages": ["unused"]}
            self.fail("consumed more than the bounded prefix")

        first = self.root / "first"
        audit = build_split(rows(), self.held_out, first, candidate_limit=7, train_rows=3, dev_rows=1)
        rerun = build_split(rows(), self.held_out, self.root / "second", candidate_limit=7, train_rows=3, dev_rows=1)
        self.assertEqual(audit["splits"], rerun["splits"])
        self.assertEqual(audit["input"]["rows"], 7)
        self.assertEqual(audit["excluded"], {"benchmark_source": 0, "held_out_exact": 1,
                                             "teacher_duplicate": 1, "above_limit": 1})

        def problems(split):
            records = map(json.loads, (first / "split" / f"{split}.jsonl").read_text().splitlines())
            result = set()
            for row in records:
                self.assertNotIn("messages", row)
                result.add(row["problem"])
            return result

        train, dev = problems("train"), problems("validation")
        self.assertEqual((len(train), len(dev)), (3, 1))
        self.assertFalse(train & dev)
        self.assertTrue((train | dev) <= set("ABCDE"))

    def test_refuses_to_treat_short_split_as_success(self):
        rows = ({"problem": letter, "solution": "1", "source": "olympiads"} for letter in "ABC")
        with self.assertRaisesRegex(ValueError, "insufficient eligible Numina rows"):
            build_split(rows, self.held_out, self.root / "short", candidate_limit=7, train_rows=3, dev_rows=1)


if __name__ == "__main__":
    unittest.main()
