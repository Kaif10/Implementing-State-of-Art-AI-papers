"""Offline evaluation-split checks; no API key or retriever required."""
import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ircot_under_test", ROOT / "main.py")
ircot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ircot
spec.loader.exec_module(ircot)


class SplitTests(unittest.TestCase):
    def test_bundled_questions_are_disjoint_from_every_demo(self):
        dev = ircot.read_jsonl(str(ROOT / "dev.jsonl"))
        for path in (ROOT / "demos").glob("*.json"):
            ircot.validate_eval_split(dev, ircot.load_json(str(path)))
        for row in dev:
            self.assertGreaterEqual(len(row["supporting_documents"]), 2)
            for filename in row["supporting_documents"]:
                self.assertTrue((ROOT / "data" / filename).is_file())

    def test_overlap_rejected_after_normalization(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            ircot.validate_eval_split(
                [{"question": "Where is the city?"}], [{"question": "WHERE is city!"}]
            )

    def test_empty_and_duplicate_evaluation_rejected(self):
        for rows in ([], [{"question": "?"}], [{"question": "X"}, {"question": "x"}]):
            with self.assertRaises(ValueError):
                ircot.validate_eval_split(rows, [])

    def test_tuning_rejects_overlap_before_client_creation(self):
        with patch.object(ircot, "read_jsonl", return_value=[{"question": "same"}]), \
             patch.object(ircot, "load_json", return_value=[{"question": "Same?"}]), \
             patch.object(ircot, "OpenAI") as client:
            with self.assertRaisesRegex(ValueError, "overlap"):
                ircot.tune_km("unused", "unused", "unused", "unused", 5)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
