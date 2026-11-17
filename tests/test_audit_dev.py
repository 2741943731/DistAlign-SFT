import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.audit_dev import prepare_audit


class AuditDevTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.input = self.root / "dev.jsonl"
        rows = [
            {"id": "a", "problem": "one", "source": "s", "solution": r"result \boxed{\frac{1}{2}}"},
            {"id": "b", "problem": "two", "source": "s", "solution": r"result \boxed{x+1}"},
            {"id": "c", "problem": "three", "source": "s", "solution": r"result \boxed{3}" + r" correction \boxed{"},
        ]
        self.input.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_extraction_and_deterministic_review_are_marked_unverified(self):
        digest = hashlib.sha256(self.input.read_bytes()).hexdigest()
        first, second = self.root / "first", self.root / "second"
        report = prepare_audit(self.input, first, expected_sha256=digest, sample_size=2)
        prepare_audit(self.input, second, expected_sha256=digest, sample_size=2)
        self.assertEqual(report["counts"], {"boxed_numeric": 1, "boxed_other": 1, "no_complete_box": 1})
        self.assertEqual((first / "review-sample.jsonl").read_bytes(), (second / "review-sample.jsonl").read_bytes())
        rows = [json.loads(line) for line in (first / "all-candidates.jsonl").read_text().splitlines()]
        self.assertIn("教师解答（未经核验）", (first / "review-sample.txt").read_text())
        self.assertTrue(all(row["manual_review_status"] == "not_reviewed" for row in rows))
        self.assertEqual(rows[0]["answer_candidate"], r"\frac{1}{2}")
        self.assertIsNone(rows[2]["answer_candidate"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_audit(self.input, first)

    def test_digest_mismatch_creates_no_output(self):
        output = self.root / "nope"
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            prepare_audit(self.input, output, expected_sha256="0" * 64)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
