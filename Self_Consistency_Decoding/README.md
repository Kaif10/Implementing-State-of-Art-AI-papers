# Self-Consistency Decoding (Paper Implementation)

This folder contains a notebook-style implementation of the Self-Consistency paper. It samples multiple chain-of-thought (CoT) paths, parses the final answers, and picks the majority vote.

## What matches the paper

- Sample many reasoning paths with stochastic decoding (temperature, top-k/top-p)
- Parse final answers from each path
- Majority vote to select the final prediction
- Separate parsing for arithmetic vs commonsense tasks

## Differences / caveats

- Implemented as a single Jupyter notebook for clarity
- Uses local model inference (not the exact model from the paper)
- No evaluation harness; this is a runnable demo

## How to run

1) Open the notebook

```
Self_Consistency_Decoding/self_consistency_decoding.ipynb
```

2) Run cells top-to-bottom. Adjust:

- `GenCfg`: number of samples `m`, temperature, and sampling params
- `TASK`: `arithmetic` or `commonsense`
- `question`: the prompt to solve

## How it works (high level)

- Build a few-shot CoT prompt
- Sample `m` completions with stochastic decoding
- Parse the answer string from each completion
- Majority vote over parsed answers

## Files

- `self_consistency_decoding.ipynb`: full implementation and demos
