import re
from typing import List
from .llm import count_tokens


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents if sents else [text]


def chunk_text(x: str, q: str, k: int, worker_instruction_template: str, enc) -> List[str]:
    """Algorithm 2 (paper): greedy sentence chunking under token budget.

    Paper pseudocode uses a static budget:
      B = k - tokens(q) - tokens(Iw)

    Implementation nuance : the actual worker prompt also includes `prev_cu`,
    which grows with the chain. To avoid context overflow in later workers, callers should
    reserve a fixed safety buffer for `prev_cu` tokens when computing B.
    """
    sentences = split_sentences(x)
    prev_cu_token_buffer = getattr(chunk_text, "prev_cu_token_buffer", 0)  # backward compat if monkeypatched
    # Note: buffer is applied outside the paper pseudocode to reserve room for growing prev_cu.
    B = (
        k
        - count_tokens(q, enc)
        - count_tokens(worker_instruction_template, enc)
        - int(prev_cu_token_buffer or 0)
    )
    if B <= 0:
        raise ValueError(
            "Token budget B<=0 after reserving safety buffer. Increase k, lower the buffer, "
            f"or shorten query/worker prompt. (k={k}, buffer={int(prev_cu_token_buffer or 0)}, B={B})"
        )

    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if count_tokens(cur, enc) + count_tokens(s, enc) > B:
            if cur:
                chunks.append(cur)
            cur = ""
        cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks
