"""Record real local-model demos and embed their GIFs in individual READMEs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "CAMEL_Paper": ("CAMEL: task specification and role-playing agents", "demo.py"),
    "Multiagent_Debate": ("Multiagent debate: independent answers and revision", "demo.py"),
    "IRCOT-multihop": ("IRCoT: interleave reasoning with retrieval", "demo.py"),
    "Self_Consistency_Decoding": ("Self-consistency: sample reasoning paths and vote", "demo.py"),
    "Program_of_Thoughts": ("Program of Thoughts: generate and execute Python", "demo.py"),
    "Speculative_Decoding": ("Speculative decoding: cached output agreement", "demo.py"),
}


def embed(project):
    folder = ROOT / project
    if not (folder / "media/demo.gif").is_file():
        raise FileNotFoundError("The inline recording must exist before updating a README")
    path = folder / "README.md"
    source = path.read_text(encoding="utf-8")
    if "<!-- project-working-demo -->" in source:
        return
    import json
    recording = json.loads((folder / "media/demo-recording.json").read_text(encoding="utf-8"))
    lines = "\n".join(event["line"] for event in recording["events"])
    checkpoint = "Qwen2.5-1.5B-Instruct" if "Qwen/Qwen2.5-1.5B-Instruct" in lines else "Qwen2.5-0.5B-Instruct"
    mode = f"Live local {checkpoint} run on CPU"
    if "dynamic int8 linear layers" in lines:
        mode += " with dynamic int8 linear layers"
    if project == "Speculative_Decoding":
        mode += "; same-checkpoint correctness demonstration"
    elif project == "Self-RAG":
        mode += "; local mean token embeddings"
    elif project == "SkillOpt_Paper":
        mode += "; uppercase-output toy task"
    elif project == "S1_Test_Time_Scaling":
        mode = "Live local DeepSeek-R1-Distill-Qwen-1.5B run on CPU with dynamic int8 linear layers"
    section = ("\n<!-- project-working-demo -->\n## Working demo\n\n"
               f"![{PROJECTS[project][0]} — actual recorded run](media/demo.gif)\n\n"
               f"{mode}. The animation plays here automatically; idle time is shortened.\n\n"
               "Install the local demo dependencies with `python -m pip install -r ../scripts/local-demo-requirements.txt`, "
               f"then run `python {PROJECTS[project][1]}` from this folder. "
               "The full output and recording details are saved in `media/`.\n")
    first, remainder = source.split("\n", 1)
    path.write_text(first+"\n"+section+"\n"+remainder, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", nargs="+", choices=list(PROJECTS), default=list(PROJECTS))
    parser.add_argument("--embed-existing", action="store_true")
    parser.add_argument("--allow-local-models", action="store_true",
                        help="Explicitly opt into RAM/CPU-intensive local inference")
    args = parser.parse_args()
    if not args.embed_existing and not args.allow_local_models:
        parser.error("Local model execution is disabled by default. Use --embed-existing to update saved recordings, or explicitly opt in with --allow-local-models.")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TRANSFORMERS_VERBOSITY="error",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    failures = []
    for project in args.projects:
        title, command = PROJECTS[project]
        print(f"\nRECORDING {project}", flush=True)
        if not args.embed_existing:
            result = subprocess.run([sys.executable, str(ROOT/"scripts/record_project_demo.py"),
                "--project", project, "--title", title, "--mode", "Live local model / actual recorded execution / CPU",
                "--timeout", "1200", "--", sys.executable, command], cwd=ROOT, env=env)
            if result.returncode:
                failures.append(project)
                continue
        embed(project)
    if failures:
        raise SystemExit("Recordings to inspect: "+", ".join(failures))


if __name__ == "__main__":
    main()
