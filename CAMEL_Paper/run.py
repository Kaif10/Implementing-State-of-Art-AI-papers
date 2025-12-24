from __future__ import annotations

import argparse
import os

from agents import OpenAIChatClient
from session import CamelRolePlay, AiSocietyConfig, CodeConfig, TerminationConfig

API_KEY = "<API_KEY>"
MODEL = "gpt-4o-mini"


def main() -> None:
    p = argparse.ArgumentParser(description="CAMEL core runner (minimal args)")
    p.add_argument("--mode", choices=["ai_society", "code"], default="ai_society")
    p.add_argument("--task", required=True)
    p.add_argument("--max-messages", type=int, default=40)
    p.add_argument("--word-limit", type=int, default=50)

    p.add_argument("--assistant-role", default="Python Programmer")
    p.add_argument("--user-role", default="Stock Trader")

    p.add_argument("--domain", default="Finance")
    p.add_argument("--language", default="Python")

    args = p.parse_args()

    model = OpenAIChatClient(model=MODEL, api_key=API_KEY)
    term = TerminationConfig(
        max_messages=args.max_messages,
        user_no_instruct_rounds=3,
        token_limit=None,
        token_model=None,
    )

    if args.mode == "ai_society":
        cfg = AiSocietyConfig(
            assistant_role=args.assistant_role,
            user_role=args.user_role,
            preliminary_task=args.task,
            word_limit=args.word_limit,
        )
        sess = CamelRolePlay.create_ai_society(model, cfg, term)
    else:
        cfg = CodeConfig(
            domain=args.domain,
            language=args.language,
            preliminary_task=args.task,
            word_limit=args.word_limit,
        )
        sess = CamelRolePlay.create_code(model, cfg, term)

    print("\n=== SPECIFIED TASK ===\n")
    print(sess.specified_task)

    def printer(u: str, a: str) -> None:
        print("\n" + "=" * 12 + " USER " + "=" * 12)
        print(u)
        print("\n" + "=" * 10 + " ASSISTANT " + "=" * 10)
        print(a)

    sess.run(on_step=printer)


if __name__ == "__main__":
    main()
