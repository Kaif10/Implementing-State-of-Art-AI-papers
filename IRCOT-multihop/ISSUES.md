# Implementation Issues and Shortcomings

This document outlines the issues found in the IRCOT implementation.

## Critical Issues

### 1. Incorrect OpenAI API Call (Line 90)
**Issue**: The code uses `client.responses.create()` which is not the standard OpenAI Python SDK API.

**Current Code**:
```python
r = client.responses.create(
    model=model,
    input=prompt,
    temperature=temp,
    max_output_tokens=max_tokens,
)
```

**Problem**: The standard OpenAI Python SDK uses `client.chat.completions.create()` with a different parameter structure.

**Expected Fix**:
```python
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=temp,
    max_tokens=max_tokens,
)
return (r.choices[0].message.content or "").strip()
```

**Note**: This might be intentional if using a custom wrapper, but it will fail with the standard OpenAI SDK.

### 2. Invalid Default Model Name (Lines 284, 291)
**Issue**: Default model is set to "gpt-5" which doesn't exist.

**Fixed**: Changed to "gpt-4o-mini" as a reasonable default.

## Potential Issues

### 3. Missing Error Handling
- No error handling for API failures
- No retry logic for transient failures
- No validation of API responses
- No handling for empty or malformed responses

### 4. Hardcoded Parameters
- `max_steps=8` is hardcoded in `ircot_answer()` function
- `doc_cap=15` is hardcoded
- `max_ctx_chars=12000` is hardcoded
- These should be configurable via command-line arguments

### 5. Answer Extraction Logic
**Issue**: The answer extraction uses simple string matching:
```python
if "answer is" in Ti.lower():
    return Ti.split(":", 1)[1].strip() if ":" in Ti else Ti.strip()
```

**Problems**:
- Very fragile - fails if model doesn't use exact phrase "answer is"
- No validation that extracted answer is reasonable
- Final reader pass uses regex but intermediate steps use simple string matching

### 6. Retrieval State Management
**Issue**: The retriever's `similarity_top_k` is modified directly:
```python
retriever.similarity_top_k = K
```

**Problem**: This mutates the retriever object, which could cause issues if the same retriever is used for multiple questions concurrently.

### 7. Demo Construction Randomness
**Issue**: Uses `random.Random()` but also calls `random.sample()` and `random.shuffle()`:
```python
rng = random.Random(seed)
chosen = rng.sample(examples, k=min(n_demos, len(examples)))
# ...
local = random.Random(seed * 10_000 + i)
distractors = local.sample(pool, k=M)
local.shuffle(docs)
```

**Problem**: Mixing seeded random with module-level random could lead to non-reproducible results.

### 8. No Progress Feedback
- Long-running operations (tuning, building demos) provide no progress indicators
- No way to resume interrupted operations
- No logging of intermediate steps

### 9. Limited Evaluation Metrics
**Issue**: Only uses Exact Match (EM) for evaluation.

**Missing**:
- F1 score
- Token-level accuracy
- Retrieval quality metrics (precision, recall)
- Reasoning step quality assessment

### 10. Context Window Management
**Issue**: Simple character-based truncation:
```python
if n + len(chunk) > max_chars:
    break
```

**Problems**:
- Doesn't account for token limits (different models have different tokenizers)
- May truncate important information arbitrarily
- No prioritization of which documents to keep

### 11. No Validation of Input Data
- No validation that demo files have required fields
- No validation that questions are non-empty
- No validation that data_dir contains readable files
- No handling of malformed JSONL files

### 12. Memory Efficiency
- Loads all documents into memory at once
- No streaming for large datasets
- Could be problematic for very large knowledge bases

### 13. Missing Dependencies Documentation
- No `requirements.txt` file
- No version pinning for dependencies
- Could lead to compatibility issues

### 14. No Unit Tests
- No test coverage
- No validation that core functions work correctly
- Difficult to verify correctness after changes

## Recommendations

1. **Fix API calls**: Update to use standard OpenAI SDK or document custom wrapper
2. **Add error handling**: Wrap API calls in try-except blocks with retries
3. **Make parameters configurable**: Add CLI arguments for hardcoded values
4. **Improve answer extraction**: Use more robust parsing or LLM-based extraction
5. **Add logging**: Implement proper logging for debugging and monitoring
6. **Add tests**: Create unit tests for core functions
7. **Add requirements.txt**: Document dependencies with versions
8. **Improve evaluation**: Add more comprehensive metrics
9. **Add progress indicators**: Show progress for long operations
10. **Validate inputs**: Add input validation and error messages
