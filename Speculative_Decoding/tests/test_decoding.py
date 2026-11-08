"""Offline algorithm checks: real tensors/caches, no model downloads."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speculative_decoding as sd
from experiment import predicted_speedup


class TinyCausalModel:
    """Prefix-dependent logits; rejected tokens left in cache change predictions."""
    def __init__(self, mode="target", probabilities=None):
        self.mode = mode
        self.probabilities = probabilities

    def __call__(self, ids, past_key_values=None, use_cache=False):
        keys = ids.float()[:, None, :, None]
        if past_key_values is not None:
            keys, _ = past_key_values.update(keys, keys, 0)
        sums = keys[:, 0, :, 0].long().cumsum(-1)[:, -ids.shape[1]:]
        if self.probabilities is not None:
            logits = torch.tensor(self.probabilities).log().expand(1, ids.shape[1], -1)
        else:
            preferred = (sums + 1) % 5
            if self.mode == "reject":
                preferred = (preferred + 1) % 5
            elif self.mode == "partial":
                preferred = (preferred + (sums % 3 == 0).long()) % 5
            logits = torch.full((*preferred.shape, 5), -20.0)
            logits.scatter_(-1, preferred.unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


@pytest.mark.parametrize("prompt", [[[1]], [[2, 1, 3]]])
@pytest.mark.parametrize("K", [1, 3, 5])
@pytest.mark.parametrize("mode", ["target", "reject", "partial"])
def test_cached_and_reference_greedy_equivalence(prompt, K, mode):
    ids = torch.tensor(prompt)
    target, draft = TinyCausalModel(), TinyCausalModel(mode)
    reference, _ = sd.autoregressive_generate(target, ids, 11, temperature=0)
    cached, _ = sd.autoregressive_generate_cached(target, ids, 11, temperature=0)
    assert torch.equal(reference, cached)
    for generate in (sd.speculative_generate, sd.speculative_generate_cached):
        actual, stats = generate(target, draft, ids, 11, K=K, temperature=0)
        assert torch.equal(actual, reference)
        if mode == "target":
            assert stats["acceptance_rate"] == 1
        elif mode == "reject":
            assert stats["accepted"] == 0
        else:
            assert 0 < stats["accepted"] < stats["proposed"]


@pytest.mark.parametrize("generate", [sd.speculative_generate, sd.speculative_generate_cached])
def test_sampling_matches_known_target(generate):
    torch.manual_seed(42)
    target = TinyCausalModel(probabilities=[0.2, 0.8])
    draft = TinyCausalModel(probabilities=[0.85, 0.15])
    hits = 0
    n = 1600
    for _ in range(n):
        ids, _ = generate(target, draft, torch.tensor([[0]]), 1, K=2)
        hits += ids[0, -1].item()
    # More than five standard deviations at p=.8; catches missing residual correction.
    assert abs(hits / n - 0.8) < 0.055


@pytest.mark.parametrize("generate", [sd.speculative_generate, sd.speculative_generate_cached])
def test_boundaries(generate):
    model = TinyCausalModel()
    ids = torch.tensor([[1]])
    result, stats = generate(model, model, ids, 0)
    assert torch.equal(result, ids) and stats["target_forwards"] == 0
    result, _ = generate(model, model, ids, 1, K=5, temperature=0)
    assert result.shape == (1, 2)
    for invalid_k in [0, -1]:
        with pytest.raises(ValueError):
            generate(model, model, ids, 1, K=invalid_k)
    with pytest.raises(ValueError):
        generate(model, model, torch.empty((1, 0), dtype=torch.long), 1)
    with pytest.raises(ValueError):
        generate(model, model, ids.repeat(2, 1), 1)
    with pytest.raises(ValueError):
        generate(model, model, ids, -1)


def test_probability_precision_and_configuration():
    logits = torch.tensor([[0., 1., 2.]], dtype=torch.float16)
    p = sd.logits_to_probs(logits, 1, top_k=1)
    assert p.dtype == torch.float32
    assert p.tolist() == [[0., 0., 1.]]
    for temperature in [-1, float("nan")]:
        with pytest.raises(ValueError):
            sd.logits_to_probs(logits, temperature)
    with pytest.raises(ValueError):
        sd.logits_to_probs(logits, 1, top_k=4)


def test_speedup_estimate_handles_perfect_acceptance():
    assert predicted_speedup(3, 1, 0) == 4
    assert predicted_speedup(3, 1, 0.25, extra_draft_steps=1) == 2


def test_timing_synchronizes_before_and_after(monkeypatch):
    import benchmark
    calls = []
    monkeypatch.setattr(benchmark, "synchronize", lambda device: calls.append("sync"))
    result, elapsed = benchmark.timed(lambda: calls.append("forward") or 42, "cpu")
    assert calls == ["sync", "forward", "sync"]
    assert result == 42 and elapsed >= 0


def test_equal_vocab_sizes_are_not_sufficient():
    from benchmark import validate_tokenizers
    target = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    other = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0})
    validate_tokenizers(target, target)
    with pytest.raises(ValueError, match="mappings"):
        validate_tokenizers(target, other)


def test_random_transformer_cache_matches_reference():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(7)
    config = GPT2Config(vocab_size=17, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    target = GPT2LMHeadModel(config).eval()
    draft = GPT2LMHeadModel(config).eval()
    for prompt in ([[1]], [[1, 2, 3]]):
        ids = torch.tensor(prompt)
        reference, _ = sd.autoregressive_generate(target, ids, 8, temperature=0)
        cached, _ = sd.autoregressive_generate_cached(target, ids, 8, temperature=0)
        speculative, _ = sd.speculative_generate_cached(target, draft, ids, 8, K=3, temperature=0)
        assert torch.equal(reference, cached)
        assert torch.equal(reference, speculative)
