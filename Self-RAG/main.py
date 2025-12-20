from __future__ import annotations

import argparse
import os

from openai import OpenAI

from kb_index import FaissKB
from selfrag import SelfRAG


# Do NOT paste real keys into code. Prefer environment variables.
# PowerShell:
#   $env:OPENAI_API_KEY="sk-..."
# cmd.exe:
#   set OPENAI_API_KEY=sk-...
OPENAI_API_KEY_FALLBACK = "<API_KEY>"


def make_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY_FALLBACK
    return OpenAI(api_key=key) if key else OpenAI()


def cmd_ingest(args: argparse.Namespace) -> None:
    client = make_client()
    FaissKB.build(
        client=client,
        docs_dir=args.docs,
        out_dir=args.out,
        embed_model=args.embed_model,
        embed_dimensions=args.embed_dimensions,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )


def cmd_ask(args: argparse.Namespace) -> None:
    client = make_client()
    kb = FaissKB.load(client=client, index_dir=args.index)

    rag = SelfRAG(
        client=client,
        kb=kb,
        gen_model=args.model,
        top_k=args.top_k,
        per_step_passages=args.per_step_passages,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )

    print(rag.answer(args.question, beam_size=args.beam_size))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("--docs", required=True)
    ing.add_argument("--out", required=True)
    ing.add_argument("--chunk-chars", type=int, default=3000)
    ing.add_argument("--overlap-chars", type=int, default=300)
    ing.add_argument("--embed-model", type=str, default="text-embedding-3-large")
    ing.add_argument("--embed-dimensions", type=int, default=1024)
    ing.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask")
    ask.add_argument("--index", required=True)
    ask.add_argument("--model", type=str, default="gpt-4o-mini")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--per-step-passages", type=int, default=3)
    ask.add_argument("--max-steps", type=int, default=10)
    ask.add_argument("--beam-size", type=int, default=1)      # 1 = greedy, >1 = segment beam
    ask.add_argument("--temperature", type=float, default=0.0) # greedy-ish
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
