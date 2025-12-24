# Implementing SOTA AI Papers

This repo is a collection of small, runnable implementations of recent AI papers (RAG, reasoning, multi-agent, etc.).
Each folder corresponds to one paper and aims to capture the core algorithmic ideas with practical, minimal code.

## Structure

- One folder per paper (e.g., `Self-RAG`, `Chain_of_Agents`, `ReACT_Paper`)
- Each folder includes its own `README.md` with paper-specific notes, differences, and how to run
- Scripts are kept simple and hackable rather than production-ready

## What to expect

- Faithful algorithmic flow where possible
- Explicit caveats when a paper uses finetuning, logprobs, or special tooling
- Minimal dependencies and clear entry points

## Getting started

1) Pick a paper folder
2) Read that folder's `README.md`
3) Run the provided CLI scripts

## Notes

- Some projects use OpenAI models; set `OPENAI_API_KEY` as needed
- These are research-style implementations, not benchmarks
