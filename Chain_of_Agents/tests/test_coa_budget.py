import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coa.chunking import chunk_text
from coa.coa import _fit_manager_prompt, run_single_path

class Encoder:
    def encode(self, text): return list(text)
    def decode(self, tokens): return "".join(tokens)

@pytest.mark.parametrize("text", ["x" * 101, "????" * 30, "One. Two. Three."])
def test_chunks_respect_budget_and_preserve_text(text):
    chunks = chunk_text(text, "", 10, "", Encoder())
    assert all(0 < len(c) <= 10 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

def test_buffers_are_per_call():
    enc = Encoder()
    assert max(map(len, chunk_text("x" * 20, "", 20, "", enc, prev_cu_token_buffer=10))) <= 10
    assert chunk_text("x" * 20, "", 20, "", enc) == ["x" * 20]

def test_manager_trims_or_rejects_before_call():
    kwargs = dict(manager_template="{task_requirements}{final_cu}{query}", query="?", enc=Encoder(), k=10, max_final_cu_tokens=100)
    prompt, cu = _fit_manager_prompt(task_requirements="Task", final_cu="x" * 100, **kwargs)
    assert len(prompt) <= 10
    with pytest.raises(ValueError, match="instructions"):
        _fit_manager_prompt(task_requirements="x" * 20, final_cu="", **kwargs)

def test_worker_budget_checked_before_api(monkeypatch):
    import coa.coa as core
    calls = []
    monkeypatch.setattr(core, "generate", lambda *a, **kw: calls.append(a) or "x" * 500)
    with pytest.raises(ValueError, match="Worker prompt"):
        run_single_path(object(), model="fake", enc=Encoder(), x="x" * 1500,
                        task_requirements="summarize", k=600,
                        prev_cu_token_buffer=0, max_prev_cu_tokens=500)
    assert len(calls) == 1
