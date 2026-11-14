import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data.prepare import prepare


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def teacher(problem, source="olympiads", **fields):
    return {"problem": problem, "solution": "The result is 1.", "source": source, **fields}


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.teacher = self.root / "teacher.jsonl"
        self.held_out = self.root / "benchmark.jsonl"
        write_rows(self.held_out, [{"problem": "Équation  1"}])

    def test_filtered_reproducible_splits_and_audit(self):
        write_rows(self.teacher, [
            teacher("Équation 1", "OpenAI/GSM8K"),
            teacher("discard by source", "MATH"),
            teacher("E\u0301quation\n1"),
            teacher("distinct question a", "orca_math", id=71),
            teacher("distinct    question a", id=72),
            teacher("distinct question b"),
            teacher("distinct question c"),
        ])
        first, second = self.root / "first", self.root / "second"
        audit = prepare(self.teacher, [self.held_out], first, seed=7)
        self.assertEqual(audit, prepare(self.teacher, [self.held_out], second, seed=7))
        self.assertEqual(audit["input"]["sha256"], hashlib.sha256(self.teacher.read_bytes()).hexdigest())
        self.assertEqual(audit["excluded"], {"benchmark_source": 2, "held_out_exact": 1,
                                             "teacher_duplicate": 1, "above_limit": 0})
        accepted = {}
        for name, count in (("train", 2), ("validation", 1)):
            payload = (first / f"{name}.jsonl").read_bytes()
            self.assertEqual(payload, (second / f"{name}.jsonl").read_bytes())
            self.assertEqual(audit["splits"][name]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(audit["splits"][name]["rows"], count)
            for line in payload.splitlines():
                row = json.loads(line)
                self.assertNotIn(row["problem_sha256"], accepted)
                accepted[row["problem_sha256"]] = row
        by_problem = {row["problem"]: row for row in accepted.values()}
        self.assertEqual(set(by_problem), {"distinct question a", "distinct question b", "distinct question c"})
        self.assertEqual(by_problem["distinct question a"]["id"], 71)
        self.assertEqual(by_problem["distinct question b"]["id"], by_problem["distinct question b"]["problem_sha256"])

    def test_capped_splits_and_refuses_overwrite(self):
        write_rows(self.teacher, [teacher(f"question {i}") for i in range(6)])
        output = self.root / "output"
        audit = prepare(self.teacher, [self.held_out], output, limit=2, validation_fraction=0.99)
        self.assertEqual(audit["excluded"]["above_limit"], 4)
        self.assertEqual(audit["splits"]["train"]["rows"], 1)
        self.assertEqual(audit["splits"]["validation"]["rows"], 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.teacher, [self.held_out], output)

    def test_invalid_teacher_leaves_no_output(self):
        write_rows(self.teacher, [teacher("good question"), {"problem": "missing solution", "source": "olympiads"}])
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, r"teacher.jsonl:2: 'solution'"):
            prepare(self.teacher, [self.held_out], output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
