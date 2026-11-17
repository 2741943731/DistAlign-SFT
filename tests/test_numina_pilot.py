import json
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from data.numina_pilot import NUMINA_COLUMNS, NUMINA_REVISION, build_split, check_local_shard, stream_teacher


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

    def test_local_shard_sha256_detects_corruption(self):
        shard = self.root / "fixture.parquet"
        shard.write_bytes(b"fixture shard")
        expected = hashlib.sha256(shard.read_bytes()).hexdigest()
        with patch("data.numina_pilot.NUMINA_SHARD_SHA256", expected):
            self.assertEqual(check_local_shard(shard), expected)
            shard.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                check_local_shard(shard)

    def test_stream_routes_local_path_or_pinned_online_url(self):
        calls = []
        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return iter([{"problem": "x"}])
        fake = types.SimpleNamespace(__version__="4.4.1", load_dataset=fake_load_dataset)
        with patch.dict(sys.modules, {"datasets": fake}):
            shard = self.root / "shard.parquet"
            self.assertEqual(list(stream_teacher(shard)), [{"problem": "x"}])
            self.assertEqual(list(stream_teacher()), [{"problem": "x"}])
        self.assertEqual(calls[0][1]["data_files"], {"train": str(shard.resolve())})
        self.assertIn(NUMINA_REVISION, calls[1][1]["data_files"]["train"])
        for args, kwargs in calls:
            self.assertEqual(args, ("parquet",))
            self.assertTrue(kwargs["streaming"])
            self.assertEqual(kwargs["columns"], list(NUMINA_COLUMNS))


if __name__ == "__main__":
    unittest.main()
