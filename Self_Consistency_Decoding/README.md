# Self-Consistency Decoding

Paper: [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

## Algorithmic flow (paper)

1) Build a few-shot chain-of-thought prompt for the task.
2) Sample multiple reasoning paths with stochastic decoding.
3) Extract the final answer from each sampled path.
4) Take the majority vote over answers to select the final prediction.

## Exceptions in this repo

- The original paper uses Google's PALM, we have used QWEN

## Run the code

1) Run cells top-to-bottom. Adjust:

- `GenCfg`: number of samples `m`, temperature, and sampling params
- `TASK`: `arithmetic` or `commonsense`
- `question`: the prompt to solve

