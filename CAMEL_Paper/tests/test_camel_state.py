import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents
from agents import ChatAgent, OpenAIChatClient, ChatMessage
from session import CamelRolePlay, TerminationConfig, parse_instruction

class Model:
    def __init__(self, replies): self.replies = iter(replies)
    def complete(self, messages): return next(self.replies)

def test_client_reused(monkeypatch):
    created = []
    def factory(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]))))
    monkeypatch.setattr(agents, "OpenAI", factory)
    client = OpenAIChatClient("fake")
    for _ in range(2): assert client.complete([ChatMessage("user", "hi")]) == "ok"
    assert len(created) == 1

def test_done_stops_without_extra_assistant_call():
    user = ChatAgent(Model(["Instruction: Solve\nInput: None", "<CAMEL_TASK_DONE>"]), "user")
    assistant = ChatAgent(Model(["Done"]), "assistant")
    session = CamelRolePlay(assistant, user, "task", TerminationConfig())
    assert len(session.run()) == 1
    assert assistant.history[-1].content == "Done"

def test_invalid_instructions_stop_at_limit():
    user = ChatAgent(Model(["hello"] * 3), "user")
    assistant = ChatAgent(Model(["reply"] * 2), "assistant")
    session = CamelRolePlay(assistant, user, "task", TerminationConfig())
    assert len(session.run()) == 2
    assert parse_instruction("Instruction: Solve\nInput: None").input == "None"
