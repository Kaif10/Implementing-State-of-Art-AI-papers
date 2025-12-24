# Chain-of-Agents (CoA) — Minimal Core Method Implementation

This repo is a compact implementation of the **core methodology** from the paper:
**"Chain of Agents: Large Language Models Collaborating on Long-Context Tasks" (NeurIPS 2024)**

Scope: **core logic only** (Algorithm 1 + Algorithm 2 + optional multi-path variants).  
Not included: paper datasets, metrics, baselines, or experimental reproduction.

---

## What matches the paper (core methodology)

- **Algorithm 1 (CoA chain):** sequential worker chain producing Communication Units (CUs), then a manager produces the final answer.
- **Algorithm 2 (chunking):** greedy sentence-wise chunking under a token budget `k`.
- **Ordering strategies:** `ltr`, `rtl`, `perm`.
- **Multi-path variants (paper Section 5.6):** `bidir`, `perm5`, `self5`.
- **Selection:** `vote` (majority vote) and optional `judge`.

Project structure:
- `coa/chunking.py`: Algorithm 2
- `coa/coa.py`: Algorithm 1 + multipath orchestration
- `coa/prompts.py`: worker/manager prompts from paper tables
- `coa/llm.py`: thin wrapper around OpenAI Responses API + token counting
- `run.py`: CLI entrypoint / wiring

---

## Important paper underspecifications (implementation choices)

These are places where the paper is **not fully specified**, so this repo makes explicit choices:

### 1) Sentence splitting
Algorithm 2 says "split into sentences" but does not specify how.  
This repo uses a simple regex splitter (`(?<=[.!?])\s+`). If you care about tricky punctuation, abbreviations, or lists, swap in a stronger sentence tokenizer.

### 2) Algorithm 2 budget vs `prev_cu` (CU growth)
The worker prompt includes `prev_cu` (the previous CU), but Algorithm 2’s budget in the paper subtracts only `tokens(q)` and `tokens(Iw)` from `k`.

This repo:
- Implements Algorithm 2’s published budget **by default**.
- Relies on `--max_output_tokens` to keep CUs from exploding (practical safeguard).
- Provides an optional **token buffer** hook in `coa/chunking.py` to reserve room for `prev_cu` if you want more robustness:
  - You can set `chunk_text.prev_cu_token_buffer = <int>` before running chunking.
  - Note: reserving a buffer is a robustness choice beyond the paper pseudocode.

### 3) Judge selector prompt
The paper mentions judge-style selection but does not provide a canonical judge prompt.
This repo includes a **minimal** judge prompt as a placeholder. Treat it as configurable, not “paper-canonical”.

---

## SECURITY: API keys (do not publish keys)

**Do not hardcode API keys into code or commit them to git.**

Recommended:
- Set `OPENAI_API_KEY` as an environment variable and use `client = OpenAI()`.

If you already committed a key:
1) Revoke/rotate it immediately.
2) Remove it from git history before publishing.

---

## Install

```bash
python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate
pip install -r requirements.txt
