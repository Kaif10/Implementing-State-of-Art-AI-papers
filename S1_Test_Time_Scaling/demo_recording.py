"""Small live budget-forcing example using the project's reasoning model."""
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo_quantization import quantize_cpu_linears
from budget_forcing import BudgetForcer


def main():
    question = "What is 12 times 13? Give the numeric answer."
    print("s1 budget forcing / live DeepSeek-R1-Distill-Qwen-1.5B / CPU / dynamic int8 linear layers", flush=True)
    print(f"Question: {question}", flush=True)
    forcer = BudgetForcer(dtype=torch.bfloat16)
    print("Preparing dynamic int8 linear layers for CPU inference...", flush=True)
    forcer.model = quantize_cpu_linears(forcer.model)
    for ignores in (0, 1):
        print(f"\nRUN / num_ignore={ignores} / thinking cap=192 tokens", flush=True)
        result = forcer.generate(question, max_thinking_tokens=192, num_ignore=ignores,
                                 answer_max_tokens=40, verbose=True)
        print(f"THINKING:\n{result.thinking}\nANSWER:\n{result.answer}", flush=True)
        print(f"Thinking tokens: {result.thinking_tokens}\nWaits injected: {result.waits_injected}\nBudget cap reached: {result.capped}", flush=True)
        if not result.answer.strip():
            raise SystemExit("No final answer was generated")
    print("DEMO COMPLETE / observed behavior for this question and model", flush=True)


if __name__ == "__main__":
    main()
