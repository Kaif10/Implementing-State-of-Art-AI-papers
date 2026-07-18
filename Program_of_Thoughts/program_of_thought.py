"""
Program of Thoughts (PoT) — minimal, honest end-to-end implementation.

Reference: Chen et al., 2022, "Program of Thoughts Prompting: Disentangling
Computation from Reasoning for Numerical Reasoning Tasks" (https://arxiv.org/abs/2211.12588).

Core idea: instead of asking the LLM to do arithmetic itself (Chain of Thought),
we ask it to *write a Python program* that computes the answer, then we run that
program in a real Python interpreter. The LLM does the reasoning; the interpreter
does the computation. This script does exactly that, end to end, and reports the
true accuracy with no massaging of the numbers.

Usage:
    # Quick single-question demo (downloads the model on first run):
    python program_of_thought.py --demo

    # Evaluate on N GSM8K test questions:
    python program_of_thought.py --num-samples 50

    # Use the 3B model instead of the 1.5B default:
    python program_of_thought.py --model Qwen/Qwen2.5-3B-Instruct --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import ast
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ----------------------------------------------------------------------------
# 1. The PoT prompt: few-shot examples that teach the model to emit Python whose
#    final answer is stored in a variable called `answer`. These are the canonical
#    GSM8K-style demonstrations used in the PoT / CoT literature.
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert at solving math word problems by writing Python code. "
    "For each question, write a short Python program that computes the answer. "
    "Do NOT do the arithmetic in your head — let Python compute it. "
    "Store the final numeric result in a variable named `answer`. "
    "Respond with a single ```python code block and nothing else."
)

FEWSHOT = [
    (
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?",
        "money_initial = 23\n"
        "bagels = 5\n"
        "bagel_cost = 3\n"
        "money_spent = bagels * bagel_cost\n"
        "answer = money_initial - money_spent",
    ),
    (
        "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. "
        "On Wednesday, he lost 2 more. How many golf balls did he have at "
        "the end of Wednesday?",
        "golf_balls_initial = 58\n"
        "golf_balls_lost_tuesday = 23\n"
        "golf_balls_lost_wednesday = 2\n"
        "answer = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from Monday to Thursday. How many computers are now "
        "in the server room?",
        "computers_initial = 9\n"
        "computers_per_day = 5\n"
        "num_days = 4  # Monday through Thursday\n"
        "computers_added = computers_per_day * num_days\n"
        "answer = computers_initial + computers_added",
    ),
]


def build_messages(question: str) -> list[dict]:
    """Build the chat message list: system + few-shot turns + the real question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, code in FEWSHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
    messages.append({"role": "user", "content": question})
    return messages


# ----------------------------------------------------------------------------
# 2. Extract the Python code from the model's response.
# ----------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> Optional[str]:
    """Pull the first fenced code block; fall back to raw text if none found."""
    m = _CODE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    # Some models forget the fences. If the text looks like code, use it as-is.
    if "answer" in text and "=" in text:
        return text.strip()
    return None


# ----------------------------------------------------------------------------
# 3. Execute the generated program in a real interpreter, in a subprocess with a
#    timeout. We append a line that prints `answer`. This runs MODEL-GENERATED
#    CODE — see the warning in the README. We isolate to a subprocess + timeout;
#    for untrusted use you should run this inside a container/VM.
# ----------------------------------------------------------------------------

_ANS_TAG = "___POT_ANSWER___"


def last_assigned_name(code: str) -> Optional[str]:
    """Return the name of the last top-level assignment target in `code`.

    The model usually stores its result in a descriptively-named variable
    (e.g. `total_cost`) and doesn't always add `answer = ...`. Reading that
    final value is part of the PoT method; which variable holds it is a harness
    detail, so we recover it here as a fallback after `answer`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    name = None
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) \
                and isinstance(node.target, ast.Name):
            name = node.target.id
    return name


def execute_code(code: str, timeout: float = 10.0) -> tuple[Optional[str], str]:
    """Run `code` and return (printed_answer_or_None, status).

    status is one of: "ok", "no_answer_var", "error", "timeout".
    Prints `answer` if defined, otherwise the last top-level assigned variable.
    """
    candidates = ["answer"]
    fallback = last_assigned_name(code)
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    selector = (
        f"\n\nfor _n in {candidates!r}:\n"
        f"    if _n in dir():\n"
        f"        print('{_ANS_TAG}', eval(_n)); break\n"
    )
    program = code + selector
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(path)

    if proc.returncode != 0:
        return None, "error"

    for line in proc.stdout.splitlines():
        if line.startswith(_ANS_TAG):
            return line[len(_ANS_TAG):].strip(), "ok"
    return None, "no_answer_var"


# ----------------------------------------------------------------------------
# 4. Numeric answer parsing / comparison.
# ----------------------------------------------------------------------------

def to_number(x: object) -> Optional[float]:
    """Best-effort parse of a value into a float. Returns None if impossible."""
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace("$", "").replace("%", "")
    # Grab the first number-looking token (handles "18.0", "-3", "42 dollars").
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def is_correct(pred: Optional[str], gold: str, tol: float = 1e-3) -> bool:
    p, g = to_number(pred), to_number(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K stores the gold answer after '####'."""
    return answer_field.split("####")[-1].strip()


# ----------------------------------------------------------------------------
# 5. Model wrapper.
# ----------------------------------------------------------------------------

class PoTModel:
    def __init__(self, model_name: str, temperature: float, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32

        print(f"[setup] loading {model_name} on {self.device} ({dtype}) ...",
              flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def generate(self, question: str) -> str:
        torch = self._torch
        messages = build_messages(question)
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True)


# ----------------------------------------------------------------------------
# 6. End-to-end: generate -> extract -> execute -> compare.
# ----------------------------------------------------------------------------

@dataclass
class Result:
    question: str
    gold: str
    raw_output: str
    code: Optional[str]
    pred: Optional[str]
    exec_status: str
    correct: bool
    gen_seconds: float


def solve(model: PoTModel, question: str, gold: str, exec_timeout: float) -> Result:
    t0 = time.time()
    raw = model.generate(question)
    gen_seconds = time.time() - t0
    code = extract_code(raw)
    if code is None:
        return Result(question, gold, raw, None, None, "no_code", False, gen_seconds)
    pred, status = execute_code(code, timeout=exec_timeout)
    correct = status == "ok" and is_correct(pred, gold)
    return Result(question, gold, raw, code, pred, status, correct, gen_seconds)


def load_gsm8k(n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(n, len(ds))
    return [(ds[i]["question"], gsm8k_gold(ds[i]["answer"])) for i in range(n)]


DEMO_QUESTION = (
    "A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "If each bolt of blue fiber costs $3 and white fiber costs $2 per bolt, "
    "and there is a 10% discount on the total, what is the final cost in dollars?"
)


def main():
    ap = argparse.ArgumentParser(description="Program of Thoughts — minimal end-to-end.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HF model id (e.g. Qwen/Qwen2.5-1.5B-Instruct or -3B-).")
    ap.add_argument("--num-samples", type=int, default=20,
                    help="Number of GSM8K test questions to evaluate.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy decoding (deterministic).")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--exec-timeout", type=float, default=10.0,
                    help="Seconds before a generated program is killed.")
    ap.add_argument("--output", default="results.jsonl",
                    help="Per-example results are written here as JSONL.")
    ap.add_argument("--demo", action="store_true",
                    help="Run a single hard-coded demo question and exit.")
    args = ap.parse_args()

    model = PoTModel(args.model, args.temperature, args.max_new_tokens)

    if args.demo:
        print(f"\n[demo] Q: {DEMO_QUESTION}\n")
        r = solve(model, DEMO_QUESTION, gold="7.2", exec_timeout=args.exec_timeout)
        print("--- generated code ---")
        print(r.code or "(no code extracted)")
        print("----------------------")
        print(f"executed -> answer={r.pred!r}  status={r.exec_status}")
        print(f"expected -> 7.2   correct={r.correct}")
        return

    print(f"[eval] loading {args.num_samples} GSM8K test questions ...", flush=True)
    data = load_gsm8k(args.num_samples)

    results: list[Result] = []
    correct = 0
    status_counts: dict[str, int] = {}
    with open(args.output, "w") as fout:
        for i, (q, gold) in enumerate(data, 1):
            r = solve(model, q, gold, exec_timeout=args.exec_timeout)
            results.append(r)
            correct += int(r.correct)
            status_counts[r.exec_status] = status_counts.get(r.exec_status, 0) + 1
            fout.write(json.dumps(asdict(r)) + "\n")
            fout.flush()
            mark = "OK " if r.correct else "XX "
            print(f"[{i:>3}/{len(data)}] {mark} pred={str(r.pred):>10} "
                  f"gold={gold:>8} status={r.exec_status:<13} "
                  f"({r.gen_seconds:4.1f}s)", flush=True)

    n = len(results)
    print("\n" + "=" * 60)
    print(f"Model:      {args.model}")
    print(f"Dataset:    GSM8K test  (n={n})")
    print(f"Decoding:   {'greedy' if args.temperature == 0 else f'sampled T={args.temperature}'}")
    print(f"Accuracy:   {correct}/{n} = {100.0 * correct / n:.1f}%")
    print(f"Exec breakdown: {status_counts}")
    print(f"Per-example results written to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
