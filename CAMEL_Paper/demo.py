"""Run the existing CAMEL role-play loop with a live local checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_demo_backend import LocalDemoClient
from session import CamelRolePlay, CodeConfig, TerminationConfig


def main():
    client = LocalDemoClient()
    class ChatModel:
        def complete(self, messages):
            return client.generate([{"role": m.role, "content": m.content} for m in messages], 160)[0]
    print("CAMEL / live task specification and role-playing agents", flush=True)
    session = CamelRolePlay.create_code(ChatModel(), CodeConfig(
        domain="Basic arithmetic", language="Python", preliminary_task="Write a Python function add(a, b) and show add(2, 3)."),
        TerminationConfig(max_messages=6))
    print("SPECIFIED TASK:\n"+session.specified_task, flush=True)
    turns = session.run(on_step=lambda user, assistant: print(
        f"\nUSER AGENT:\n{user}\nASSISTANT AGENT:\n{assistant}", flush=True))
    if not turns:
        raise SystemExit("No role-playing exchange completed")
    print(f"DEMO COMPLETE / {len(turns)} completed exchanges", flush=True)


if __name__ == "__main__":
    main()
