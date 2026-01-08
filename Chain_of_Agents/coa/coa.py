# coa/coa.py
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .chunking import chunk_text
from .llm import count_tokens, generate
from . import prompts as P



# This module is intentionally split into two parts:

#   1) CORE (Algorithm 1): run_single_path()
#      - the worker chain builds CU_1..CU_l, then a manager synthesizes the answer
#      - this is the "main" CoA algorithm

#   2) OPTIONAL PAPER EXTRAS (Section 5.6): run_multipath()
#      - multi-path variants (bidir / perm5 / self5)
#      - selection via majority vote or a judge model

# I have kept a hard visual separation so readers don't confuse the extras as required for the core algorithm.

# Small utilities (used by both core and extras)


def _trim_keep_head_tail(text: str, enc, max_tokens: int) -> str:
    """
    Trim to <= max_tokens while preserving BOTH the start and end of the text.
    This is a practical default for CUs:
      - Start often contains the "final answer" or key claims (workers may rewrite CU with answer first)
      - End contains the most recent updates
    """
    text = (text or "").strip()
    if not text or max_tokens <= 0:
        return ""
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text

    head = max_tokens // 2
    tail = max_tokens - head
    kept = toks[:head] + toks[-tail:]
    return enc.decode(kept).strip()


def _choose_order_indices(n: int, order: str, seed: int) -> List[int]:
    idxs = list(range(n))
    if order == "ltr":
        return idxs
    if order == "rtl":
        return list(reversed(idxs))
    if order == "perm":
        rnd = random.Random(seed)
        rnd.shuffle(idxs)
        return idxs
    raise ValueError("order must be one of: ltr, rtl, perm")


def _fit_manager_prompt(
    manager_template: str,
    *,
    task_requirements: str,
    final_cu: str,
    query: str,
    enc,
    k: int,
    max_final_cu_tokens: int,
) -> Tuple[str, str]:
    """
    Ensure the manager prompt fits within k tokens by trimming final_cu if needed.
    Returns: (prompt, trimmed_final_cu)
    """
    cu = (final_cu or "").strip()
    cu = _trim_keep_head_tail(cu, enc, max_final_cu_tokens)

    prompt = manager_template.format(
        task_requirements=task_requirements.strip(),
        final_cu=cu,
        query=query,
    )

    # If still too long (e.g., huge task_requirements), trim CU further.
    # Keep it simple and deterministic.
    for _ in range(5):
        if count_tokens(prompt, enc) <= k:
            return prompt, cu
        # Remove another 20% of CU tokens each iteration.
        cur_tokens = len(enc.encode(cu))
        new_tokens = max(64, int(cur_tokens * 0.8))
        if new_tokens >= cur_tokens:
            break
        cu = _trim_keep_head_tail(cu, enc, new_tokens)
        prompt = manager_template.format(
            task_requirements=task_requirements.strip(),
            final_cu=cu,
            query=query,
        )

    # Last resort: return whatever we have; API may still reject if requirements are massive.
    return prompt, cu



# 1) CORE CoA — Algorithm 1 (Single Path)


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
    order: str = "ltr",   # ltr | rtl | perm  (used in ablations; core defaults to ltr)
    seed: int = 0,
    retries: int = 0,
    # Practical robustness knobs (not in paper, but required in real systems):
    prev_cu_token_buffer: int = 1024,
    max_prev_cu_tokens: int = 1024,
    max_final_cu_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    CORE CoA (Algorithm 1): worker chain -> manager.

    Paper formulation:
      Split x into chunks {c1..cl} where each ci < k
      CU0 = ""
      For i in 1..l:
          CUi = LLM_W(IW, CUi-1, ci, q)
      Response = LLM_M(IM, CU_l, q)

    This implementation:
      - uses chunk_text() (Algorithm 2) for greedy sentence chunking under budget
      - reserves prev_cu_token_buffer tokens during chunking to avoid context overflow
      - trims prev_cu to max_prev_cu_tokens to keep prompts stable

    Returns:
      - chunks_used
      - communication_units (CU0..CUl)
      - final_cu
      - answer
    """
    query = (query or "")
    has_query = bool(query.strip())

    worker_template = P.WORKER_QUERY if has_query else P.WORKER_NONQUERY
    manager_template = P.MANAGER_QUERY if has_query else P.MANAGER_NONQUERY

    # ---- Stage 0: chunking under a safe budget ----
    # chunk_text() supports reserving a safety buffer for prev_cu via a function attribute.
    # This avoids later worker prompts exceeding the model context window.
    setattr(chunk_text, "prev_cu_token_buffer", int(prev_cu_token_buffer))

    chunks = chunk_text(
        x=x,
        q=query,
        k=k,
        worker_instruction_template=worker_template,
        enc=enc,
    )
    idxs = _choose_order_indices(len(chunks), order=order, seed=seed)

    # ---- Stage 1: worker chain ----
    cu_prev = ""
    cus: List[str] = [""]  # CU0
    chunks_used: List[str] = []

    for step_i, chunk_idx in enumerate(idxs, start=1):
        chunk = chunks[chunk_idx]
        chunks_used.append(chunk)

        prev_cu = _trim_keep_head_tail(cu_prev, enc, int(max_prev_cu_tokens))

        worker_prompt = worker_template.format(
            i=step_i,
            chunk=chunk,
            prev_cu=prev_cu,
            query=query,
        )

        cu_i = generate(
            client,
            model,
            worker_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retries=retries,
        )

        cus.append(cu_i)
        cu_prev = cu_i

    final_cu = cu_prev

    # ---- Stage 2: manager ----
    # Default manager CU cap: keep room for requirements/query + template.
    if max_final_cu_tokens is None:
        # heuristic: allow most of k to be CU, but keep some headroom
        max_final_cu_tokens = max(512, int(k * 0.75))

    manager_prompt, final_cu_trimmed = _fit_manager_prompt(
        manager_template,
        task_requirements=task_requirements,
        final_cu=final_cu,
        query=query,
        enc=enc,
        k=k,
        max_final_cu_tokens=int(max_final_cu_tokens),
    )

    answer = generate(
        client,
        model,
        manager_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
    )

    return {
        "chunks_used": chunks_used,
        "communication_units": cus,
        "final_cu": final_cu_trimmed,
        "answer": answer,
    }



# 2) OPTIONAL PAPER EXTRAS — Multi-path CoA (Section 5.6)


def majority_vote(answers: List[str]) -> str:
    """
    Majority vote over answer strings.
    Deterministic tie-break: earliest appearance among the max-frequency answers.
    """
    normalized = [(a or "").strip() for a in answers]
    counts: Dict[str, int] = {}
    for a in normalized:
        counts[a] = counts.get(a, 0) + 1

    best = ""
    best_n = -1
    for a in normalized:
        n = counts.get(a, 0)
        if n > best_n:
            best, best_n = a, n
    return best


def _judge_select_and_answer(
    client: OpenAI,
    *,
    judge_model: str,
    task_requirements: str,
    query: str,
    candidate_cus: List[Tuple[str, str]],  # (path_id, final_cu)
    max_output_tokens: int,
    temperature: float,
    retries: int,
) -> str:
    """
    Paper: w/ judge uses an LLM to judge the most reliable CU_l among paths and generate the final answer.
    We implement this with a simple judge prompt (repo default in prompts.py).
    """
    candidates_txt = "\n\n".join([f"{cid}:\n{cu}" for cid, cu in candidate_cus])
    prompt = P.JUDGE_PROMPT.format(
        task_requirements=task_requirements.strip(),
        query=(query or "").strip(),
        candidates=candidates_txt,
    )

    out = generate(
        client,
        judge_model,
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retries=retries,
    ).strip()

    # Expected format:
    #   path_3
    #   <final answer...>
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
    # Keep robustness knobs consistent with single-path:
    prev_cu_token_buffer: int = 1024,
    max_prev_cu_tokens: int = 1024,
    max_final_cu_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    OPTIONAL: Multi-path CoA (paper Section 5.6).

    The paper explores multiple "paths" (multiple CoA runs) and selects the best result via:
      - w/ vote: majority vote over final answers
      - w/ judge: LLM judges the most reliable final CU and answers from it

    Paths supported here (matching paper terminology):
      - bidir: 2-way (left-to-right, right-to-left)
      - perm5: 5-way permutation of chunk order
      - self5: 5-way self-consistency (same order, temperature > 0)

    Returns:
      - path_answers
      - path_final_cus
      - final
    """
    path_answers: List[str] = []
    path_final_cus: List[str] = []

    if mode == "bidir":
        orders = ["ltr", "rtl"]
        for j, ord_ in enumerate(orders):
            r = run_single_path(
                client,
                model=model,
                enc=enc,
                x=x,
                task_requirements=task_requirements,
                query=query,
                k=k,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                order=ord_,
                seed=seed + j,
                retries=retries,
                prev_cu_token_buffer=prev_cu_token_buffer,
                max_prev_cu_tokens=max_prev_cu_tokens,
                max_final_cu_tokens=max_final_cu_tokens,
            )
            path_answers.append(r["answer"])
            path_final_cus.append(r["final_cu"])

    elif mode == "perm5":
        for j in range(5):
            r = run_single_path(
                client,
                model=model,
                enc=enc,
                x=x,
                task_requirements=task_requirements,
                query=query,
                k=k,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                order="perm",
                seed=seed + j,
                retries=retries,
                prev_cu_token_buffer=prev_cu_token_buffer,
                max_prev_cu_tokens=max_prev_cu_tokens,
                max_final_cu_tokens=max_final_cu_tokens,
            )
            path_answers.append(r["answer"])
            path_final_cus.append(r["final_cu"])

    elif mode == "self5":
        if self_consistency_temperature is None:
            raise ValueError("self5 requires self_consistency_temperature (paper uses temp > 0)")
        for j in range(5):
            r = run_single_path(
                client,
                model=model,
                enc=enc,
                x=x,
                task_requirements=task_requirements,
                query=query,
                k=k,
                max_output_tokens=max_output_tokens,
                temperature=float(self_consistency_temperature),
                order="ltr",
                seed=seed + j,
                retries=retries,
                prev_cu_token_buffer=prev_cu_token_buffer,
                max_prev_cu_tokens=max_prev_cu_tokens,
                max_final_cu_tokens=max_final_cu_tokens,
            )
            path_answers.append(r["answer"])
            path_final_cus.append(r["final_cu"])

    else:
        raise ValueError("mode must be: bidir, perm5, self5")

    # ---- Select final ----
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
