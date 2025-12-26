# Self-RAG (Practical Implementation)

Paper: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

## Algorithmic flow (paper)

1) Start with a query and the current partial answer.
2) Before generating each next sentence, decide whether the need to retrieve (`RetrieveDecision`).
3) If retrieving, fetch top passages; if not, continue with the model alone.
4) For each passage, generate a candidate next sentence while emitting reflection labels:
   - `isrel` (relevance), `issup` (support), `isuse` (usefulness).
5) Score and filter candidates with hard gates/logic for irrelevant or unsupported content.
6) Append the best candidate sentence to the answer.
7) Repeat until an `is_final` stop condition or a max-step limit is reached.
8) Optionally run segment-level beam search over sentence candidates.
9) The models are finetuned to emit the reflection labels.

## Exceptions in this repo

- No finetuned model; reflection labels are generated as JSON and parsed.
- No token-level logprobs for reflection labels.
- Retrieval policy is conservative to reduce hallucinations on doc-specific questions.
- The no-context path refuses to guess policy facts.

## Run the code

1) Install deps

```bash
python -m venv .venv
. .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install openai faiss-cpu numpy pydantic pypdf
```

2) Set your API key (or put it in the root `.env`)

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

## Files

- `main.py`: CLI entry point (`ingest`, `ask`)
- `kb_index.py`: chunking + embeddings + FAISS
- `selfrag.py`: Self-RAG inference loop and scoring
- `kb/`: example knowledge base docs
