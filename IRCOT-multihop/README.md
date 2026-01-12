# IRCOT (Iterative Retrieval Chain-of-Thought) Implementation

This implementation follows the IRCOT paper approach for multi-hop question answering by interleaving retrieval steps with chain-of-thought reasoning.

## Overview

IRCOT improves multi-hop question answering by:
1. Starting with an initial retrieval based on the question
2. Iteratively generating reasoning steps (chain-of-thought)
3. Retrieving additional documents based on each reasoning step
4. Continuing until sufficient information is gathered
5. Generating a final answer

## Installation

### Option 1: Using Virtual Environment (Recommended)

**Windows PowerShell:**
```powershell
# Create and activate virtual environment
.\setup_venv.ps1
```

**Windows CMD:**
```cmd
setup_venv.bat
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Manual Installation

```bash
pip install openai llama-index llama-index-retrievers-bm25
```

**Note:** If you already have these packages installed globally (e.g., from another project), the code will work without installation. However, using a virtual environment is recommended to avoid dependency conflicts.

## Data Format

### Knowledge Base Documents
Place text files in the `data/` folder. Each file will be chunked and indexed using BM25 retrieval.

Example files have been provided:
- `wikipedia_albert_einstein.txt`
- `wikipedia_relativity.txt`
- `wikipedia_quantum_mechanics.txt`
- `wikipedia_nobel_prize.txt`
- `wikipedia_ulm.txt`

### Demo Source File (`demos_source.jsonl`)
Each line should be a JSON object with:
```json
{
  "question": "...",
  "cot": "... answer is: ...",
  "gold_docs": [{"title": "...", "text": "..."}, ...],
  "distractor_docs": [{"title": "...", "text": "..."}, ...]
}
```

### Development Set (`dev.jsonl`)
Each line should be a JSON object with:
```json
{
  "question": "...",
  "answer": "..."
}
```

## Usage

### 1. Build Demo Sets

First, create demo sets from your source file:

**Linux/Mac:**
```bash
python main.py build-demos \
  --source demos_source.jsonl \
  --out_dir demos \
  --n_demos 8 \
  --Ms 1 2 3 \
  --seeds 0 1 2
```

**Windows CMD:**
```cmd
python main.py build-demos --source demos_source.jsonl --out_dir demos --n_demos 8 --Ms 1 2 3 --seeds 0 1 2
```

**Windows PowerShell:**
```powershell
python main.py build-demos `
  --source demos_source.jsonl `
  --out_dir demos `
  --n_demos 8 `
  --Ms 1 2 3 `
  --seeds 0 1 2
```

This creates demo sets with different numbers of distractors (M) and random seeds.

### 2. Tune Hyperparameters

Find the best K (retrieval size) and M (number of distractors):

**Linux/Mac:**
```bash
python main.py tune \
  --data_dir data \
  --dev dev.jsonl \
  --demos_dir demos \
  --model gpt-4o-mini \
  --max_examples 50
```

**Windows CMD:**
```cmd
python main.py tune --data_dir data --dev dev.jsonl --demos_dir demos --model gpt-4o-mini --max_examples 50
```

**Windows PowerShell:**
```powershell
python main.py tune `
  --data_dir data `
  --dev dev.jsonl `
  --demos_dir demos `
  --model gpt-4o-mini `
  --max_examples 50
```

This will test different combinations of K and M and report the best configuration.

### 3. Answer a Question

Use the trained configuration to answer a question:

**Linux/Mac:**
```bash
python main.py answer \
  --data_dir data \
  --demos demos/demos_M2_seed0.json \
  --question "What city was Albert Einstein born in?" \
  --model gpt-4o-mini \
  --K 4
```

**Windows CMD:**
```cmd
python main.py answer --data_dir data --demos demos\demos_M2_seed0.json --question "What city was Albert Einstein born in?" --model gpt-4o-mini --K 4
```

**Windows PowerShell:**
```powershell
python main.py answer `
  --data_dir data `
  --demos demos\demos_M2_seed0.json `
  --question "What city was Albert Einstein born in?" `
  --model gpt-4o-mini `
  --K 4
```

## Testing the Algorithm

### Quick Setup Test

First, verify your setup works:

```bash
python test_ircot.py
```

This will check:
- Data files are present
- Retriever can be built
- API key is configured (if set)

For a full test including API calls:

```bash
python test_ircot.py --full
```

### Step-by-Step Testing

#### For Linux/Mac (Bash):

1. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_MODEL="gpt-4o-mini"  # or gpt-4, gpt-3.5-turbo, etc.
   ```

2. **Build demo sets**:
   ```bash
   python main.py build-demos --source demos_source.jsonl --out_dir demos
   ```

3. **Test with a single question**:
   ```bash
   python main.py answer \
     --data_dir data \
     --demos demos/demos_M1_seed0.json \
     --question "What year did Einstein receive the Nobel Prize?" \
     --model gpt-4o-mini \
     --K 4
   ```

4. **Run hyperparameter tuning** (optional):
   ```bash
   python main.py tune \
     --data_dir data \
     --dev dev.jsonl \
     --demos_dir demos \
     --model gpt-4o-mini \
     --max_examples 5
   ```

#### For Windows (CMD):

1. **Set up environment variables**:
   ```cmd
   set OPENAI_API_KEY=your-api-key
   set OPENAI_MODEL=gpt-4o-mini
   ```
   Note: These settings are temporary for the current CMD session. For permanent settings, use System Properties > Environment Variables.

2. **Build demo sets**:
   ```cmd
   python main.py build-demos --source demos_source.jsonl --out_dir demos
   ```

3. **Test with a single question** (single line):
   ```cmd
   python main.py answer --data_dir data --demos demos\demos_M1_seed0.json --question "What year did Einstein receive the Nobel Prize?" --model gpt-4o-mini --K 4
   ```
   
   Or multi-line (use `^` for line continuation):
   ```cmd
   python main.py answer ^
     --data_dir data ^
     --demos demos\demos_M1_seed0.json ^
     --question "What year did Einstein receive the Nobel Prize?" ^
     --model gpt-4o-mini ^
     --K 4
   ```

4. **Run hyperparameter tuning** (optional):
   ```cmd
   python main.py tune --data_dir data --dev dev.jsonl --demos_dir demos --model gpt-4o-mini --max_examples 5
   ```
   
   Or multi-line:
   ```cmd
   python main.py tune ^
     --data_dir data ^
     --dev dev.jsonl ^
     --demos_dir demos ^
     --model gpt-4o-mini ^
     --max_examples 5
   ```

#### For Windows PowerShell:

1. **Set up environment variables**:
   ```powershell
   $env:OPENAI_API_KEY="your-api-key"
   $env:OPENAI_MODEL="gpt-4o-mini"
   ```

2. **Build demo sets**:
   ```powershell
   python main.py build-demos --source demos_source.jsonl --out_dir demos
   ```

3. **Test with a single question**:
   ```powershell
   python main.py answer `
     --data_dir data `
     --demos demos\demos_M1_seed0.json `
     --question "What year did Einstein receive the Nobel Prize?" `
     --model gpt-4o-mini `
     --K 4
   ```

4. **Run hyperparameter tuning** (optional):
   ```powershell
   python main.py tune `
     --data_dir data `
     --dev dev.jsonl `
     --demos_dir demos `
     --model gpt-4o-mini `
     --max_examples 5
   ```

### Expected Output

The `answer` command will output the final answer extracted from the reasoning chain.

The `tune` command will output accuracy scores for different K/M combinations and identify the best configuration.

## Implementation Notes

- **Retrieval**: Uses BM25 retrieval from LlamaIndex
- **Chunking**: Documents are split into chunks of 512 characters with 20 character overlap
- **Max Steps**: Default is 8 iterative retrieval steps
- **Document Cap**: Maximum 15 documents collected per question
- **Context Limit**: Maximum 12,000 characters in context

## Known Issues and Shortcomings

**⚠️ Important**: This implementation has several issues that need to be addressed. See `ISSUES.md` for a detailed list.

**Critical Issues**:
1. **API Call Format**: Uses `client.responses.create()` which may not work with standard OpenAI SDK
2. **Missing Error Handling**: No retry logic or error handling for API failures
3. **Hardcoded Parameters**: Many parameters are hardcoded and not configurable

See `ISSUES.md` for the complete list of issues and recommendations.
