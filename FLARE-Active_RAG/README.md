# FLARE: Forward-Looking Active REtrieval-Augmented Generation

Faithful implementation of the FLARE paper (https://arxiv.org/abs/2305.06983).

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

## Implementation Notes

This is a faithful implementation of the FLARE algorithm from the paper:

1. **Forward-looking generation**: Generates temporary next sentence before finalizing
2. **Confidence-based retrieval**: Checks token probabilities; retrieves if < threshold (paper: 0.05)
3. **Sentence mask threshold**: Falls back to ratio of uncertain tokens if logprobs unavailable (paper: 0.1)
4. **FLARE-direct approach**: Uses generated sentence as retrieval query
5. **Sentence-by-sentence**: Builds answer incrementally with active retrieval

**Paper thresholds implemented**:
- `confidence_threshold = 0.05` (5% token probability threshold)
- `sentence_mask_threshold = 0.1` (10% ratio of uncertain tokens in sentence)

**Differences from paper**:
- Uses OpenAI API logprobs instead of custom model logprobs
- BM25 retrieval instead of DPR/Elasticsearch

Requires `OPENAI_API_KEY` environment variable.
