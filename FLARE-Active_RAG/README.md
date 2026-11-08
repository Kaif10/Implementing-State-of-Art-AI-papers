# FLARE: Forward-Looking Active REtrieval-Augmented Generation

API-based implementation of the FLARE retrieval loop (https://arxiv.org/abs/2305.06983).

## How FLARE Works (Paper Algorithm)

1. **Generate temporary next sentence** (forward-looking)
2. **Check token confidence** - if probability < threshold, trigger retrieval
3. **Use sentence as query** (FLARE-direct) to retrieve relevant documents
4. **Regenerate sentence** with retrieved context
5. **Append to final answer** and repeat

**Key insight**: FLARE generates a temporary sentence to look ahead, checks if it's confident, and retrieves BEFORE finalizing (not after).

## Installation

```bash
pip install openai llama-index llama-index-retrievers-bm25
```

## Setup

Set your OpenAI API key:

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-api-key"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=your-api-key
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key"
```

## Testing

### Quick Test with Sample Questions

**Windows PowerShell:**
```powershell
cd FLARE-Active_RAG
python main.py test
```

**Windows CMD:**
```cmd
cd FLARE-Active_RAG
python main.py test
```

**Linux/Mac:**
```bash
cd FLARE-Active_RAG
python main.py test
```

This will run all questions from `test_questions.jsonl` and show the answers.

### Answer a Single Question

**Windows PowerShell:**
```powershell
python main.py answer --data_dir data --question "What is quantum computing?"
```

**Windows CMD:**
```cmd
python main.py answer --data_dir data --question "What is quantum computing?"
```

**Linux/Mac:**
```bash
python main.py answer --data_dir data --question "What is quantum computing?"
```

## Parameters

### Answer Command
- `--data_dir`: Directory containing knowledge base documents (required)
- `--question`: Question to answer (required)
- `--model`: OpenAI model (default: gpt-4o-mini)
- `--top_k`: Number of documents to retrieve (default: 3)
- `--theta`: Confidence threshold θ (paper: 0.4/0.8, default 0.4)
- `--beta`: Masking threshold β (paper: 0.4, default 0.4)
- `--max_steps`: Max generation steps (default: 10)

### Test Command
- `--data_dir`: Directory containing knowledge base documents (default: data)
- `--questions`: Path to questions JSONL file (default: test_questions.jsonl)
- `--model`: OpenAI model (default: gpt-4o-mini)
- `--top_k`: Number of documents to retrieve (default: 3)
- `--theta`: Confidence threshold θ (paper: 0.4/0.8, default 0.4)
- `--beta`: Masking threshold β (paper: 0.4, default 0.4)
- `--max_steps`: Max generation steps (default: 5)

## Data Format

Place text files (`.txt` or `.md`) in the `data/` directory. Example files are provided:
- `data/science_quantum_computing.txt`
- `data/science_artificial_intelligence.txt`
- `data/history_renaissance.txt`
- `data/technology_blockchain.txt`

## Implementation notes

- Confidence is calculated only over tokens overlapping the selected first sentence.
- `theta` (default `0.4`) triggers retrieval when a sentence token falls below it.
- `beta` (default `0.4`) removes low-confidence tokens from the retrieval query.
- Token bytes preserve alignment across Unicode characters. Missing or misaligned
  logprobs trigger conservative retrieval; an empty filtered query uses the question.
- Retrieval uses local BM25. Sentence boundaries and completion detection are simple
  heuristics; the bundled question runner prints outputs for inspection.

```bash
python -m unittest discover -s tests -v
```

Offline tests need `openai`; no API key or model calls are required.
