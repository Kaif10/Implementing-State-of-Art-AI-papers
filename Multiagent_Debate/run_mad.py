# run_mad.py
from __future__ import annotations

import argparse
import os
from typing import Optional

from openai import OpenAI

from mad_core import DebateConfig, run_debate, TaskName
from mad_bio_eval import load_facts, eval_biography


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
        
from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["arithmetic", "gsm8k", "mmlu", "biography", "chess"])
    ap.add_argument("--query", default=None, help="Task input (expression/problem/question/person/moves).")
    ap.add_argument("--input_file", default=None, help="Read task input from a file instead of --query.")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2, help="TOTAL rounds including Round 1.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--other_mode", choices=["concat", "summarize"], default="concat")
    ap.add_argument("--print_agents", action="store_true")

    # Optional biography eval (paper Appendix A.2)
    ap.add_argument("--bio_facts_file", default=None, help="Newline-separated ground-truth facts (Wikipedia bullets).")
    ap.add_argument("--bio_person", default=None, help="Person name for biography eval (if task=biography).")
    ap.add_argument("--critic_model", default="gpt-4o-mini")

    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")

    if args.input_file:
        user_input = read_text_file(args.input_file)
    else:
        user_input = (args.query or "").strip()

    if not user_input:
        raise SystemExit("Provide --query or --input_file.")

    client = OpenAI()

    cfg = DebateConfig(
        model=args.model,
        agents=args.agents,
        rounds=args.rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        other_mode=args.other_mode,
    )

    result = run_debate(
        task=args.task,  # type: ignore[arg-type]
        user_input=user_input,
        cfg=cfg,
        client=client,
    )

    print("\n=== FINAL ===")
    print(result.final_text)

    if args.print_agents:
        print("\n=== AGENTS (last round) ===")
        for i, txt in enumerate(result.agent_texts, 1):
            print(f"\n[Agent {i}]")
            print(txt)

    # Optional biography eval (separate from generation)
    if args.task == "biography" and args.bio_facts_file:
        person = args.bio_person or user_input
        facts = load_facts(args.bio_facts_file)

        ev = eval_biography(
            person=person,
            generated_bio=result.agent_texts[0],  # evaluated on final generated bio text
            ground_truth_facts=facts,
            critic_model=args.critic_model,
            temperature=0.0,
            client=client,
        )

        print("\n=== BIO EVAL (LLM critic) ===")
        print(f"total_facts={ev.total_facts} yes={ev.yes} no={ev.no} uncertain={ev.uncertain}")
        print(f"decided_rate (yes+no)/total = {ev.decided_rate:.3f}")
        print(f"yes_rate_over_decided yes/(yes+no) = {ev.yes_rate_over_decided:.3f}")

if __name__ == "__main__":
    main()
