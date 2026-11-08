"""Structured critique validation and beam behavior with synthetic passages."""
import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace
from pydantic import ValidationError

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selfrag_under_test", ROOT / "selfrag.py")
rag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rag
spec.loader.exec_module(rag)


class SelfRAGTests(unittest.TestCase):
    def test_schema_rejects_unknown_labels_and_string_boolean(self):
        valid = dict(isrel="relevant", issup="fully_supported", isuse=5, is_final=True)
        for key, value in [("isrel", "maybe"), ("issup", "unknown"), ("isuse", 6), ("is_final", "false")]:
            with self.assertRaises(ValidationError):
                rag.CritiqueLabels(**(valid | {key: value}))

    def test_unknown_support_cannot_pass_gate(self):
        model = rag.SelfRAG(None, None, model="test")
        self.assertFalse(model._accept("relevant", "unknown"))

    def test_beam_score_does_not_reward_length(self):
        self.assertEqual(rag.Beam(["a"], [], 4).mean_score,
                         rag.Beam(["a", "b"], [], 8).mean_score)

    def test_rejected_passages_fall_back_in_both_modes(self):
        passage = SimpleNamespace(chunk_id="p", source="p.txt", text="irrelevant")
        for beam_size in (1, 2):
            model = rag.SelfRAG(None, SimpleNamespace(search=lambda *a: [passage]), model="test")
            model.decide_retrieve = lambda *a, **kw: rag.RetrieveDecision(decision="yes", query="q")
            model.generate_candidate = lambda *a: rag.PassageCandidate(
                sentence="Unsupported.", critique=rag.CritiqueLabels(
                    isrel="irrelevant", issup="no_support", isuse=1, is_final=True))
            model.generate_noctx = lambda *a: rag.NoCtxStep(sentence="Insufficient evidence.", isuse=1, is_final=True)
            self.assertEqual(model.answer("q", beam_size=beam_size), "Insufficient evidence.")

    def test_missing_parsed_output_is_explicit_error(self):
        client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kw: SimpleNamespace(output_parsed=None)))
        model = rag.SelfRAG(client, None, model="test")
        with self.assertRaisesRegex(ValueError, "no parsed output"):
            model._parse_with_retries(messages=[], text_format=rag.NoCtxStep)


if __name__ == "__main__":
    unittest.main()
