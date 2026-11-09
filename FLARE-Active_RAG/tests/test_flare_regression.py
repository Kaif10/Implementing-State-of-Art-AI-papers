"""Sentence probability alignment and retrieval control without model calls."""
import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flare_under_test", ROOT / "main.py")
flare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = flare
spec.loader.exec_module(flare)


class ConfidenceTests(unittest.TestCase):
    def make_client(self, responses):
        calls = []
        replies = iter(responses)
        def create(**kwargs):
            calls.append(kwargs)
            text, finish_reason = next(replies)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason=finish_reason, logprobs=None)])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls

    def test_token_limited_fragment_is_retried_before_finalization(self):
        client, calls = self.make_client([("Quantum computing uses", "length"),
                                          ("Quantum computing uses qubits.", "stop")])
        model = flare.FLARE(client, SimpleNamespace(similarity_top_k=1))
        text, _ = model._call_llm("Question", max_tokens=50)
        self.assertEqual(text, "Quantum computing uses qubits.")
        self.assertGreater(calls[1]["max_tokens"], calls[0]["max_tokens"])

    def test_complete_first_sentence_does_not_retry_a_discarded_tail(self):
        client, calls = self.make_client([("Known fact. Later unfinished", "length")])
        model = flare.FLARE(client, SimpleNamespace(similarity_top_k=1))
        text, _ = model._call_llm("Question", max_tokens=50)
        self.assertEqual(len(calls), 1)
        self.assertEqual(flare.sentence_confidence(text, None, .4)[0], "Known fact.")

    def test_repeated_truncation_is_not_silently_punctuated(self):
        client, calls = self.make_client([("Still incomplete", "length"), ("Still incomplete", "length")])
        model = flare.FLARE(client, SimpleNamespace(similarity_top_k=1))
        with self.assertRaisesRegex(ValueError, "token limit"):
            model._call_llm("Question", max_tokens=50)
        self.assertEqual(len(calls), 2)

    def test_later_uncertain_sentence_does_not_trigger_retrieval(self):
        sentence, confidence, query = flare.sentence_confidence(
            "Known fact. Guess later.", [(b"Known fact.", .9), (b" Guess later.", .01)], .4
        )
        self.assertEqual((sentence, confidence, query), ("Known fact.", .9, "Known fact."))

    def test_uncertain_tokens_removed_from_query(self):
        sentence, confidence, query = flare.sentence_confidence(
            "Born in Paris.", [(b"Born in", .9), (b" Paris", .1), (b".", .9)], .4
        )
        self.assertEqual(confidence, .1)
        self.assertNotIn("Paris", query)

    def test_unicode_bytes_and_cross_sentence_token(self):
        raw = "Café. Later.".encode()
        tokens = [(raw[:4], .9), (raw[4:6], .8), (raw[6:], .7)]
        self.assertEqual(flare.sentence_confidence("Café. Later.", tokens, .4),
                         ("Café.", .8, "Café."))

    def test_misalignment_has_no_confidence(self):
        self.assertIsNone(flare.sentence_confidence("Fact.", [(b"Other", .9)], .4)[1])

    def test_missing_probabilities_retrieve_and_regenerate(self):
        model = flare.FLARE(None, SimpleNamespace(similarity_top_k=1), max_steps=1)
        replies = iter([("Draft.", None), ("Grounded.", None)])
        model._call_llm = lambda *args, **kwargs: next(replies)
        queries = []
        model._retrieve_documents = lambda q: queries.append(q) or ["Evidence"]
        self.assertEqual(model.generate("Question?", initial_context=""), "Grounded.")
        self.assertEqual(queries, ["Draft."])

    def test_uncertainty_only_in_later_sentence_makes_no_retrieval(self):
        model = flare.FLARE(None, SimpleNamespace(similarity_top_k=1), max_steps=1)
        model._call_llm = lambda *a, **kw: ("Known. Guess.", [(b"Known.", .9), (b" Guess.", .01)])
        model._retrieve_documents = lambda q: self.fail("Unexpected retrieval")
        self.assertEqual(model.generate("Question?", initial_context=""), "Known.")


if __name__ == "__main__":
    unittest.main()
