import argparse
from pathlib import Path
from openai import OpenAI

from coa.llm import get_encoder
from coa.coa import run_single_path, run_multipath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--task_requirements_file", required=True)
    ap.add_argument("--query", default="")

    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--tokenizer_model", default="", help="tiktoken model name; defaults to --model")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--max_output_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--order", choices=["ltr", "rtl", "perm"], default="ltr")
    ap.add_argument("--multipath", choices=["none", "bidir", "perm5", "self5"], default="none")
    ap.add_argument("--selector", choices=["vote", "judge"], default="vote")
    ap.add_argument("--judge_model", default="")
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--self_consistency_temperature", type=float, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    x = Path(args.input_file).read_text(encoding="utf-8")
    req = Path(args.task_requirements_file).read_text(encoding="utf-8")

    tok_model = args.tokenizer_model.strip() or args.model
    enc = get_encoder(tok_model)
    client = OpenAI()


    if args.multipath == "none":
        r = run_single_path(
            client,
            model=args.model,
            enc=enc,
            x=x,
            task_requirements=req,
            query=args.query,
            k=args.k,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            order=args.order,
            seed=args.seed,
            retries=args.retries,
        )
        print(r["answer"])
        if args.debug:
            print("\n--- final CU ---\n")
            print(r["final_cu"])
        return

    r = run_multipath(
        client,
        model=args.model,
        enc=enc,
        x=x,
        task_requirements=req,
        query=args.query,
        k=args.k,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        mode=args.multipath,
        selector=args.selector,
        judge_model=(args.judge_model.strip() or None),
        judge_temperature=args.judge_temperature,
        self_consistency_temperature=args.self_consistency_temperature,
        seed=args.seed,
        retries=args.retries,
    )
    print(r["final"])
    if args.debug:
        print("\n--- path answers ---")
        for i, a in enumerate(r["path_answers"], start=1):
            print(f"[path_{i}] {a}")


if __name__ == "__main__":
    main()
