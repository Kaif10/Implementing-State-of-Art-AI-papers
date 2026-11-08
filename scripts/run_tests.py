"""Run each paper's offline tests in an isolated process, with no API credentials."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = [
    ("CAMEL_Paper", "tests"),
    ("Chain_of_Agents", "tests"),
    ("Multiagent_Debate", "tests"),
    ("Program_of_Thoughts", "test_scoring.py"),
    ("Reflexion_Paper", "test_judge.py"),
    ("SkillOpt_Paper", "tests"),
    ("Self_Consistency_Decoding", "test_answer_parsing.py"),
    ("IRCOT-multihop", "tests"),
    ("FLARE-Active_RAG", "tests"),
    ("Self-RAG", "tests"),
]
MODELS = [("Speculative_Decoding", "tests"), ("S1_Test_Time_Scaling", "tests")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-models", action="store_true", help="Include synthetic decoding tests (requires torch and transformers)")
    args = parser.parse_args()
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RUN_MODEL_TESTS"):
        env.pop(key, None)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    failures = []
    for project, tests in CORE + (MODELS if args.include_models else []):
        print(f"\nTesting {project}", flush=True)
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", tests, "-q", "-p", "no:cacheprovider"], cwd=ROOT / project, env=env, timeout=180)
            if result.returncode:
                failures.append(project)
        except subprocess.TimeoutExpired:
            print(f"Timed out: {project}", flush=True)
            failures.append(project)
    if failures:
        print("Failed projects: " + ", ".join(failures))
        return 1
    print("All selected project suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
