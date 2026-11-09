"""Run the notebook's self-consistency implementation as a small terminal demo.

Uses the notebook's actual classes, prompts, generation and voting functions;
model setup is explicit here so recordings do not depend on notebook UI state.
"""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from answer_parsing import ANSWER_IS, parse_arithmetic_first, parse_commonsense_first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--question", default="How many minutes are in 3.75 hours?")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    print(f"Model: {args.model} / CPU / {args.samples} sampled reasoning paths", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    scope = dict(globals(), tokenizer=tokenizer, model=model, DEVICE=torch.device("cpu"))
    notebook = json.loads(Path(__file__).with_name("self_consistency_decoding.ipynb").read_text(encoding="utf-8"))
    definitions = {"trim_after_first_answer", "CoTExample", "GenCfg", "build_prompt",
                   "majority_vote", "generate_many", "self_consistency"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("!"):
            continue
        tree = ast.parse(source)
        nodes = [node for node in tree.body if
                 isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions
                 or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARITH_EX" for t in node.targets)]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-demo-definitions", "exec"), scope)
    cfg = scope["GenCfg"](m=args.samples, batch_size=1, seed=42, max_new_tokens=80)
    print(f"Question: {args.question}", flush=True)
    final, counts, paths, parsed = scope["self_consistency"](
        args.question, examples=scope["ARITH_EX"], parser=parse_arithmetic_first, cfg=cfg)
    media = Path(__file__).with_name("media")
    media.mkdir(exist_ok=True)
    (media / "raw-samples.json").write_text(json.dumps(
        {"question": args.question, "paths": paths, "parsed": parsed, "votes": counts, "final": final},
        indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    for i, (path, answer) in enumerate(zip(paths, parsed), 1):
        marker = re.search(r"The answer is\s*[+-]?\d+(?:\.\d+)?", path, re.IGNORECASE)
        excerpt = path[:marker.end()] if marker else path
        print(f"\nPATH {i} / excerpt through first answer: {excerpt}\nPARSED ANSWER: {answer}", flush=True)
    print(f"\nVOTES: {counts}\nFINAL: {final}\nParsed paths: {sum(p is not None for p in parsed)}/{len(parsed)}", flush=True)
    print("Full generated paths: media/raw-samples.json", flush=True)
    if final is None:
        raise SystemExit("No final answer could be parsed; inspect the output")


if __name__ == "__main__":
    main()
