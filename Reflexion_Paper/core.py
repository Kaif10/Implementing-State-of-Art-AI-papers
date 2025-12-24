from dataclasses import dataclass
import json
import re
import textwrap

from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "your_key_here"

@dataclass
class Task:
    instruction: str
    context: str = ""


@dataclass
class EvalResult:
    passed: bool
    feedback: str


@dataclass
class Attempt:
    answer: str
    trajectory: str
    eval: EvalResult
    reflection: str | None = None



# LLM

class LLM:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def call(self, system: str, user: str, *, temperature=0.0, max_tokens=500, json_object=False) -> str:
        text_arg = {"format": {"type": "json_object"}} if json_object else None
        r = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            text=text_arg,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return (r.output_text or "").strip()


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _clean_json(text: str) -> str:
    s = text.strip()
    # Remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)


def parse_json_object(raw: str, required_keys: tuple[str, ...] = ()) -> dict | None:
    if not raw:
        return None
    candidates = []
    stripped = _strip_code_fences(raw)
    if stripped:
        candidates.append(stripped)
    extracted = _extract_first_json_object(stripped or raw)
    if extracted:
        candidates.append(extracted)

    for cand in candidates:
        try:
            obj = json.loads(_clean_json(cand))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if any(k not in obj for k in required_keys):
            continue
        return obj
    return None

# Roles.

# Actor 
def actor(llm: LLM, task: Task, memory: list[str]) -> tuple[str, str]:
    mem = ""
    if memory:
        mem = "REFLECTIONS (lessons from prior failed attempts):\n" + "\n".join(f"- {m}" for m in memory)
    else:
        mem = "REFLECTIONS: none"

    user = textwrap.dedent(f"""\
    {mem}

    TASK:
    {task.instruction}

    CONTEXT:
    {task.context}

    Solve the task. Treat the reflections as distilled lessons to avoid repeating mistakes,
    but do not mention them in the answer.
    First write a brief, high-level trajectory (3-7 steps). Then give the final answer.

    Output ONLY valid JSON in this schema:
    {{
      "trajectory": "1) ...\\n2) ...",
      "answer": "..."
    }}
    Rules: JSON only, no extra keys, double quotes, no trailing commas.
    """)
    raw = llm.call(
        "You are the Actor, an expert problem solver. Follow the task and constraints exactly.",
        user,
        temperature=0.7,
        max_tokens=500,
        json_object=True,
    )
    obj = parse_json_object(raw, required_keys=("answer", "trajectory"))
    if not obj:
        return raw.strip(), "PARSE_ERROR: actor JSON invalid"
    return str(obj["answer"]).strip(), str(obj["trajectory"]).strip()


#Judge (The original paper uses different hard evaluation criteria for each task; here we use a generic LLM based evaluator)
def judge(llm: LLM, task: Task, answer: str) -> EvalResult:
    user = textwrap.dedent(f"""\
    You are grading whether the CANDIDATE ANSWER fully satisfies the TASK and any constraints in the CONTEXT.

    TASK:
    {task.instruction}

    CONTEXT:
    {task.context}

    CANDIDATE ANSWER:
    {answer}

    Decide PASS/FAIL. Be strict:
    - FAIL if any required constraint is missing or violated.
    - FAIL if the answer contradicts the task or context.
    - If information is insufficient to verify a requirement, FAIL and say what is missing.
    - Do NOT invent extra requirements beyond what is stated.

    Output ONLY valid JSON exactly in this schema:
    {{
      "passed": true/false,
      "must_fix": ["...","..."],
      "feedback": "one short paragraph"
    }}
    Rules: JSON only, no extra keys, double quotes, no trailing commas.

    Notes:
    - If passed=false, must_fix must list the concrete problems to fix.
    - Be strict and specific.
    """)
    '''
    raw = llm.call(
        "You are the Evaluator. Judge only against the task and context, and output JSON only.",
        user,
        temperature=0.0,
        max_tokens=250,
    )
    '''
    raw = llm.call("You are the Evaluator. Judge only against the task and context, and output JSON only.",
        user, temperature=0.0, max_tokens=200, json_object=True)


    obj = parse_json_object(raw, required_keys=("passed",))
    if not obj:
        return EvalResult(False, "Evaluator output was not valid JSON.")
    passed = bool(obj["passed"])
    must_fix = obj.get("must_fix", [])
    feedback = obj.get("feedback", "")

    # fold must_fix into feedback so your reflector sees it even if you keep EvalResult simple
    if (not passed) and must_fix:
        feedback = feedback.strip() + "\nMust fix:\n- " + "\n- ".join(str(x) for x in must_fix)

    return EvalResult(passed, feedback)


# reflector agent
def reflect(llm: LLM, task: Task, trajectory: str, answer: str, ev: EvalResult) -> str:
    user = textwrap.dedent(f"""\
    TASK:
    {task.instruction}

    TRAJECTORY (step history):
    {trajectory}

    ANSWER:
    {answer}

    EVALUATOR FEEDBACK:
    {ev.feedback}

    Write a concise self-reflection in first person:
    - what went wrong (if anything)
    - a generalized lesson that would help on similar tasks
    - the concrete change for the next attempt
    Keep it brief (2-4 sentences).
    Output ONLY the reflection.
    """)
    return llm.call(
        "You are the Self-Reflector. Be concise, actionable, and generalize lessons.",
        user,
        temperature=0.0,
        max_tokens=160,
    )


# Entire Reflexion loop (core)

def reflexion_run(actor_llm: LLM, judge_llm: LLM, reflect_llm: LLM, task: Task, *, trials: int = 4, omega: int = 3):
    memory: list[str] = []
    attempts: list[Attempt] = []

    for t in range(1, trials + 1):
        ans, traj = actor(actor_llm, task, memory)
        ev = judge(judge_llm, task, ans)
        att = Attempt(answer=ans, trajectory=traj, eval=ev)

        if ev.passed:
            attempts.append(att)
            return True, memory, attempts

        att.reflection = reflect(reflect_llm, task, traj, ans, ev)
        memory.append(att.reflection)
        memory = memory[-omega:]
        attempts.append(att)

    return False, memory, attempts


# Demo Run

if __name__ == "__main__":
    task = Task(
        instruction=(
            "Design a 48-hour emergency response plan for a coastal city facing a major hurricane landfall.\n\n"
            "Output requirements:\n"
            "- Use exactly 8 time blocks labeled: T-24, T-18, T-12, T-6, T+0, T+6, T+12, T+18\n"
            "- For each block include: Objective, 2-3 Actions, Lead agency\n"
            "- Include at least 3 decision triggers written as IF/THEN\n"
            "- Include a resource table with totals that sum to exactly 120 units across: Medical, Shelter, Power, Transport\n"
            "- Include a communication plan with exactly 4 channels\n"
            "- End with a 3-item risk/mitigation list\n"
            "- Output ONLY the plan (no preamble)"
        )
    )

    actor_llm = LLM("gpt-4o-mini")
    judge_llm = LLM("gpt-4o-mini")
    reflect_llm = LLM("gpt-4o-mini")

    ok, memory, attempts = reflexion_run(actor_llm, judge_llm, reflect_llm, task, trials=5, omega=3)
    print(attempts[-1].answer)
