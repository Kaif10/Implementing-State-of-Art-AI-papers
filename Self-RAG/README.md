# Self-RAG (Practical Implementation)

This folder contains a minimal, working implementation of the Self-RAG paper with a few pragmatic changes:
- no finetuned model or reflection tokens/logprobs
- reflection labels are emitted via JSON parsing (`responses.parse`) instead
- a single generator handles both generation + self-evaluation (no separate critic at inference)

The goal is to keep the core behavior (retrieve/reflect/ground) while staying simple and hackable.

## What this implements from the paper

- Sentence-level retrieval decision (`RetrieveDecision`) before each next sentence
- One candidate per passage with reflection labels:
  - `isrel` (relevance), `issup` (support), `isuse` (usefulness)
- Selection by a weighted score with hard gates for unsupported/irrelevant content
- Optional segment-level beam search (`beam_size > 1`) like the paper

All of the logic lives in `selfrag.py`.

## Differences vs the paper

- No finetuned model; reflection labels are generated and parsed from JSON
- No token-level logprobs for reflection labels
- A conservative retrieve policy to reduce hallucinations on doc-specific questions
- No-context path refuses to guess policy facts

## Quick start

1) Install deps (example)

```bash
python -m venv .venv
. .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install openai faiss-cpu numpy pydantic pypdf
```

2) Set your API key

```powershell
$env:OPENAI_API_KEY="sk-..."
```

3) Build the index

```bash
python main.py ingest --docs kb --out faiss_index
```

4) Ask a question

```bash
python main.py ask --index faiss_index "What counts as restricted data?"
```

## How it works (high level)

- `kb_index.py` chunks documents and builds a FAISS index
- `selfrag.py` runs the Self-RAG loop:
  - decide retrieve
  - fetch passages
  - generate one candidate sentence per passage with reflection labels
  - score + filter candidates, append the best sentence
  - stop when `is_final` is true or `max_steps` reached

## Files

- `main.py`: CLI entry point (`ingest`, `ask`)
- `kb_index.py`: chunking + embeddings + FAISS
- `selfrag.py`: Self-RAG inference loop and scoring
- `kb/`: example knowledge base docs

## Notes

- `pypdf` is only needed if you ingest PDFs.
- The CLI defaults to small, conservative settings (greedy decode, low temperature).
  Tune `--beam-size`, `--top-k`, and `--per-step-passages` as needed.
