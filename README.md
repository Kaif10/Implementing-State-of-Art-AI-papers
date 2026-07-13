# Implementing SOTA AI Papers

This repo is a collection  implementations of some of the best and my favourite research publications in the space of AI in recent years with focus on Generative AI.
All implementations are in Python.
Each folder corresponds to one paper and aims to capture the core algorithmic ideas with practical, minimal code (I have skipped experimentations/comparisons, only focusing on the core methodology)

## Implemented papers

| Folder | Paper |
|--------|-------|
| [CAMEL_Paper](CAMEL_Paper) | [CAMEL: Communicative Agents for "Mind" Exploration of LLM Society](https://arxiv.org/abs/2303.17760) |
| [Chain_of_Agents](Chain_of_Agents) | [Chain of Agents: LLMs Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818) |
| [FLARE-Active_RAG](FLARE-Active_RAG) | [Active Retrieval Augmented Generation (FLARE)](https://arxiv.org/abs/2305.06983) |
| [IRCOT-multihop](IRCOT-multihop) | [Interleaving Retrieval with Chain-of-Thought for Multi-Step QA (IRCoT)](https://arxiv.org/abs/2212.10509) |
| [Multiagent_Debate](Multiagent_Debate) | [Improving Factuality and Reasoning through Multiagent Debate](https://arxiv.org/abs/2305.14325) |
| [Program_of_Thoughts](Program_of_Thoughts) | [Program of Thoughts Prompting](https://arxiv.org/abs/2211.12588) |
| [Reflexion_Paper](Reflexion_Paper) | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) |
| [S1_Test_Time_Scaling](S1_Test_Time_Scaling) | [s1: Simple Test-Time Scaling (budget forcing)](https://arxiv.org/abs/2501.19393) |
| [Self-RAG](Self-RAG) | [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) |
| [Self_Consistency_Decoding](Self_Consistency_Decoding) | [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) |
| [SkillOpt_Paper](SkillOpt_Paper) | [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904) |
| [Speculative_Decoding](Speculative_Decoding) | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) |

## Structure

- Each folder includes its own `README.md` with paper-specific notes, differences in the implementations, and how to run
- Scripts are kept simple and something you can implement and play around with too.

## What to expect

- Algorithmic flow as close and faithful to the original paper as possible
- You can expect explicit caveats like when a paper uses finetuning, I might have used off the shelf LLM API, but I have mentioned all caveats/exceptions I made to the original paper in the respective READMEs

