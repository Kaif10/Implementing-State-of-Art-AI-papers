"""Run the existing debate orchestration with real local model generations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_demo_backend import LocalDemoClient
from mad_core import DebateConfig, run_debate


def main():
    client = LocalDemoClient()
    question = "12 * 13"
    print("Multiagent debate / two agents / two rounds / live local model", flush=True)
    print("QUESTION: "+question, flush=True)
    result = run_debate(task="arithmetic", user_input=question,
        cfg=DebateConfig(model=client.name, agents=2, rounds=2, temperature=.7, max_tokens=120), client=client)
    for i, (text, parsed) in enumerate(zip(result.agent_texts, result.agent_parsed), 1):
        print(f"\nAGENT {i}:\n{text}\nPARSED ANSWER: {parsed}", flush=True)
    print(f"\nFINAL VOTED ANSWER: {result.final_parsed}", flush=True)
    if result.final_parsed != "156":
        raise SystemExit("The voted answer did not match 12 * 13 = 156")
    print("DEMO COMPLETE", flush=True)


if __name__ == "__main__":
    main()
