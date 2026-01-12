# ircot_end2end.py
# pip install openai llama-index llama-index-retrievers-bm25

import os, re, json, random, argparse
import warnings
from statistics import mean
from typing import Dict, List, Iterable, Tuple

# Suppress "resource module not available on Windows" warning
# This is harmless - llama-index dependencies try to import Unix-only resource module
warnings.filterwarnings("ignore", message=".*resource module.*", category=ImportWarning)

from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever

SENT_END = re.compile(r"[.?!]\s")


# ---------------- IO ----------------

def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- LlamaIndex BM25 ----------------

def build_retriever(
    data_dir: str,
    top_k: int,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> BM25Retriever:
    docs = SimpleDirectoryReader(data_dir, recursive=True).load_data()
    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(docs)
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)


# ---------------- Prompt formatting (paper template) ----------------

def first_sentence(s: str) -> str:
    s = " ".join((s or "").strip().split())
    m = SENT_END.search(s)
    return s if not m else s[: m.end() - 1].strip()


def node_text(node) -> str:
    try:
        return node.get_content(metadata_mode="none")
    except TypeError:
        return node.get_content()


def fmt_nodes(nodes: List, max_chars: int = 12000) -> str:
    out, n = [], 0
    for node in nodes:
        meta = getattr(node, "metadata", {}) or {}
        title = meta.get("file_name") or meta.get("source") or "Document"
        txt = node_text(node).strip()
        chunk = f"Wikipedia Title: {title}\n{txt}\n"
        if n + len(chunk) > max_chars:
            break
        out.append(chunk)
        n += len(chunk)
    return "\n".join(out).strip()


def fmt_demo(d: dict) -> str:
    # d: {"docs":[{"title","text"},...], "question":..., "cot":...}
    docs = "\n".join(f"Wikipedia Title: {x['title']}\n{x['text'].strip()}" for x in d["docs"])
    return f"{docs}\nQ: {d['question']}\nA: {d['cot'].strip()}\n"


# ---------------- OpenAI calls ----------------

def llm(client: OpenAI, model: str, prompt: str, temp: float, max_tokens: int) -> str:
    r = client.responses.create(
        model=model,
        input=prompt,
        temperature=temp,
        max_output_tokens=max_tokens,
    )
    return (r.output_text or "").strip()


# ---------------- IRCoT core ----------------

def ircot_answer(
    Q: str,
    retriever: BM25Retriever,
    demos: List[dict],
    client: OpenAI,
    model: str,
    K: int,
    max_steps: int = 8,
    doc_cap: int = 15,
    max_ctx_chars: int = 12000,
) -> str:
    # set per-step retrieval size
    retriever.similarity_top_k = K

    demo_block = "\n".join(fmt_demo(d) for d in demos)

    P: List = []
    seen = set()

    def add(retrieved):
        # paper: cap total collected paragraphs (stop adding when cap reached)
        nonlocal P, seen
        for nws in retrieved:
            if len(P) >= doc_cap:
                break
            n = nws.node
            nid = getattr(n, "node_id", None) or getattr(n, "id_", None)
            if nid and nid not in seen:
                seen.add(nid)
                P.append(n)

    # initial retrieval
    add(retriever.retrieve(Q))

    cot: List[str] = []

    for _ in range(max_steps):
        ctx = fmt_nodes(P, max_chars=max_ctx_chars)
        prior = " ".join(cot)
        prompt = demo_block + "\n" + f"{ctx}\nQ: {Q}\nA: {prior}".rstrip() + "\n"

        # model may generate multiple sentences; keep only the first
        Ti = first_sentence(llm(client, model, prompt, temp=0.2, max_tokens=120))
        if not Ti:
            break
        cot.append(Ti)

        if "answer is" in Ti.lower():
            return Ti.split(":", 1)[1].strip() if ":" in Ti else Ti.strip()

        add(retriever.retrieve(Ti))

    # reader pass (full CoT, then extract "answer is:")
    ctx = fmt_nodes(P, max_chars=max_ctx_chars)
    reader_prompt = demo_block + "\n" + f"{ctx}\nQ: {Q}\nA:"
    out = llm(client, model, reader_prompt, temp=0.0, max_tokens=256)
    m = re.search(r"answer is:\s*(.*)", out, flags=re.IGNORECASE)
    return (m.group(1).strip() if m else out.strip())


# ---------------- Demo construction (gold + M distractors, shuffled) ----------------

def build_demo_sets(
    source_jsonl: str,
    out_dir: str,
    n_demos: int,
    Ms: List[int],
    seeds: List[int],
) -> None:
    """
    source_jsonl lines must contain:
      {
        "question": "...",
        "cot": ".... answer is: ...",
        "gold_docs": [{"title":"...","text":"..."}, ...],
        "distractor_docs": [{"title":"...","text":"..."}, ...]
      }
    """
    examples = read_jsonl(source_jsonl)
    if not examples:
        raise ValueError("No examples in source_jsonl")

    os.makedirs(out_dir, exist_ok=True)

    for M in Ms:
        for seed in seeds:
            rng = random.Random(seed)
            chosen = rng.sample(examples, k=min(n_demos, len(examples)))

            demos = []
            for i, ex in enumerate(chosen):
                gold = list(ex["gold_docs"])
                pool = list(ex["distractor_docs"])
                if len(pool) < M:
                    raise ValueError(f"Example has {len(pool)} distractors but M={M}")

                local = random.Random(seed * 10_000 + i)
                distractors = local.sample(pool, k=M)

                docs = gold + distractors
                local.shuffle(docs)  # shuffled + concatenated

                demos.append({"docs": docs, "question": ex["question"], "cot": ex["cot"]})

            write_json(os.path.join(out_dir, f"demos_M{M}_seed{seed}.json"), demos)
            print(f"wrote {out_dir}/demos_M{M}_seed{seed}.json")


# ---------------- K/M sweep (avg over 3 demo sets) ----------------

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> int:
    return int(norm(pred) == norm(gold))


def tune_km(
    data_dir: str,
    dev_jsonl: str,
    demos_dir: str,
    model: str,
    max_examples: int,
) -> None:
    dev = read_jsonl(dev_jsonl)
    client = OpenAI()

    Ks = [2, 4, 6, 8]
    Ms = [1, 2, 3]
    seeds = [0, 1, 2]  # paper: average over 3 demo sets

    best = (-1.0, None)

    for K in Ks:
        retriever = build_retriever(data_dir, top_k=K)
        for M in Ms:
            scores = []
            for seed in seeds:
                demos = load_json(os.path.join(demos_dir, f"demos_M{M}_seed{seed}.json"))

                hits, n = 0, 0
                for ex in dev[:max_examples]:
                    pred = ircot_answer(
                        Q=ex["question"],
                        retriever=retriever,
                        demos=demos,
                        client=client,
                        model=model,
                        K=K,
                    )
                    hits += em(pred, ex["answer"])
                    n += 1
                scores.append(hits / max(1, n))

            avg = mean(scores)
            print(f"K={K} M={M} seed_scores={['%.3f' % s for s in scores]} avg={avg:.3f}")
            if avg > best[0]:
                best = (avg, {"K": K, "M": M, "seed_scores": scores})

    print("\nBEST:", {"avg": best[0], **(best[1] or {})})


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-demos")
    p1.add_argument("--source", required=True, help="demos_source.jsonl")
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--n_demos", type=int, default=8)
    p1.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 3])
    p1.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    p2 = sub.add_parser("tune")
    p2.add_argument("--data_dir", required=True)
    p2.add_argument("--dev", required=True, help="dev.jsonl with {question, answer}")
    p2.add_argument("--demos_dir", required=True)
    p2.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p2.add_argument("--max_examples", type=int, default=50)

    p3 = sub.add_parser("answer")
    p3.add_argument("--data_dir", required=True)
    p3.add_argument("--demos", required=True, help="path to demos_M{M}_seed{seed}.json")
    p3.add_argument("--question", required=True)
    p3.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p3.add_argument("--K", type=int, default=4)

    args = ap.parse_args()

    if args.cmd == "build-demos":
        build_demo_sets(args.source, args.out_dir, args.n_demos, args.Ms, args.seeds)

    elif args.cmd == "tune":
        tune_km(args.data_dir, args.dev, args.demos_dir, args.model, args.max_examples)

    elif args.cmd == "answer":
        client = OpenAI()
        retriever = build_retriever(args.data_dir, top_k=args.K)
        demos = load_json(args.demos)
        ans = ircot_answer(args.question, retriever, demos, client, args.model, K=args.K)
        print(ans)


if __name__ == "__main__":
    main()
