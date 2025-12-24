import random
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI

from .chunking import chunk_text
from .llm import generate
from . import prompts as P


def majority_vote(answers: List[str]) -> str:
    counts: Dict[str, int] = {}
    for a in answers:
        a = (a or "").strip()
        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return ""
    # deterministic tie-break: earliest among max
    best = None
    best_n = -1
    for a in answers:
        a = (a or "").strip()
        n = counts[a]
        if n > best_n:
            best = a
            best_n = n
    return best or ""


def run_single_path(
    client: OpenAI,
    *,
    model: str,
    enc,
    x: str,
    task_requirements: str,
    query: str = "",
    k: int = 8192,
    max_output_tokens: int = 1024,
    temperature: float = 0.0,
    order: str = "ltr",   # ltr | rtl | perm
    seed: int = 0,
    retries: int = 0,
) -> Dict[str, Any]:
    """Algorithm 1 (paper): worker chain -> manager.

    Returns:
      - chunks_used
      - communication_units (CU0..)
      - final_cu
      - answer
    """
    is_query = bool((query or "").strip())
    worker_template = P.WORKER_QUERY if is_query else P.WORKER_NONQUERY
    manager_template = P.MANAGER_QUERY if is_query else P.MANAGER_NONQUERY

    chunks = chunk_text(x=x, q=query, k=k, worker_instruction_template=worker_template, enc=enc)
    idxs = list(range(len(chunks)))

    if order == "rtl":
        idxs = list(reversed(idxs))
    elif order == "perm":
        rnd = random.Random(seed)
        rnd.shuffle(idxs)
    elif order != "ltr":
        raise ValueError("order must be: ltr, rtl, perm")

    cu_prev = ""
    cus: List[str] = [""]  # CU0
    chunks_used: List[str] = []

    # Stage 1: workers
    for i, ci in enumerate(idxs, start=1):
        chunk = chunks[ci]
        chunks_used.append(chunk)

        prompt = worker_template.format(
            i=i,
            chunk=chunk,
            prev_cu=cu_prev,
            query=query,
        )
        cu_i = generate(
            client, model, prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )
        cus.append(cu_i)
        cu_prev = cu_i

    final_cu = cu_prev

    # Stage 2: manager
    manager_prompt = manager_template.format(
        task_requirements=task_requirements.strip(),
        final_cu=final_cu,
        query=query,
    )
    answer = generate(
        client, model, manager_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
    )

    return {
        "chunks_used": chunks_used,
        "communication_units": cus,
        "final_cu": final_cu,
        "answer": answer,
    }


def _judge_select_and_answer(
    client: OpenAI,
    *,
    judge_model: str,
    task_requirements: str,
    query: str,
    candidate_cus: List[Tuple[str, str]],  # (id, final_cu)
    max_output_tokens: int,
    temperature: float,
    retries: int,
) -> str:
    candidates_txt = "\n\n".join([f"{cid}:\n{cu}" for cid, cu in candidate_cus])
    prompt = P.JUDGE_PROMPT.format(
        task_requirements=task_requirements.strip(),
        query=(query or "").strip(),
        candidates=candidates_txt,
    )
    out = generate(
        client, judge_model, prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
    )
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("path_"):
        return "\n".join(lines[1:]).strip()
    return out


def run_multipath(
    client: OpenAI,
    *,
    model: str,
    enc,
    x: str,
    task_requirements: str,
    query: str = "",
    k: int = 8192,
    max_output_tokens: int = 1024,
    temperature: float = 0.0,
    mode: str = "bidir",          # bidir | perm5 | self5
    selector: str = "vote",       # vote | judge
    judge_model: Optional[str] = None,
    judge_temperature: float = 0.0,
    self_consistency_temperature: Optional[float] = None,
    seed: int = 0,
    retries: int = 0,
) -> Dict[str, Any]:
    """Paper multi-path variants (Section 5.6)."""
    path_answers: List[str] = []
    path_final_cus: List[str] = []

    if mode == "bidir":
        orders = ["ltr", "rtl"]
        for j, order in enumerate(orders):
            r = run_single_path(
                client, model=model, enc=enc, x=x, task_requirements=task_requirements, query=query,
                k=k, max_output_tokens=max_output_tokens, temperature=temperature,
                order=order, seed=seed + j, retries=retries,
            )
            path_answers.append(r["answer"])
            path_final_cus.append(r["final_cu"])

    elif mode == "perm5":
        for j in range(5):
            r = run_single_path(
                client, model=model, enc=enc, x=x, task_requirements=task_requirements, query=query,
                k=k, max_output_tokens=max_output_tokens, temperature=temperature,
                order="perm", seed=seed + j, retries=retries,
            )
            path_answers.append(r["answer"])
            path_final_cus.append(r["final_cu"])

    elif mode == "self5":
        if self_consistency_temperature is None:
            raise ValueError("self5 requires self_consistency_temperature")
        for j in range(5):
            r = run_single_path(
                client, model=model, enc=enc, x=x, task_requirements=task_requirements, query=query,
                k=k, max_output_tokens=max_output_tokens, temperature=float(self_consistency_temperature),
                order="ltr", seed=seed + j, retries=retries,
            )
            path_answers.append(r["answer"])
            path_final_cus.append(r["final_cu"])

    else:
        raise ValueError("mode must be: bidir, perm5, self5")

    if selector == "vote":
        final = majority_vote(path_answers)
    elif selector == "judge":
        jm = judge_model or model
        candidates = [(f"path_{i+1}", cu) for i, cu in enumerate(path_final_cus)]
        final = _judge_select_and_answer(
            client,
            judge_model=jm,
            task_requirements=task_requirements,
            query=query,
            candidate_cus=candidates,
            max_output_tokens=max_output_tokens,
            temperature=judge_temperature,
            retries=retries,
        )
    else:
        raise ValueError("selector must be: vote, judge")

    return {
        "path_answers": path_answers,
        "path_final_cus": path_final_cus,
        "final": final,
    }
