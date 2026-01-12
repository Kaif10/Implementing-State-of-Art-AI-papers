# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

An inspired implementation of [Self-RAG](https://arxiv.org/abs/2310.11511) using off-the-shelf LLM APIs.

## Paper Methodology

This implementation follows **Algorithm 1** and **Appendix A.3** as closely as possible:


# Self-RAG: Retrieve + Generate + Critique (Self-Reflection)

An implementation inspired by the paper **Self-RAG (Learning to Retrieve, Generate, and Critique through Self-Reflection)** using off-the-shelf LLM APIs.

## Paper methodology (in plain English)

Self-RAG is just **RAG with a loop**:

* The model **decides whether retrieval is needed**
* If it retrieves, it generates **one candidate next sentence per retrieved chunk**
* The model also **self-grades** each candidate (relevance + support + usefulness)
* It **picks the best candidate** and appends it to the answer
* Repeat until the answer is done

---

## Core loop (one “segment” = one sentence)

Repeat until `is_final = true` (or max steps reached):

### 1) Retrieval decision

Given:

* user prompt `x`
* answer so far `y` y=0 (nothing) at start

The model outputs:

`Retrieve ∈ {yes, no, continue}`

* **yes** = fetch new chunks from the KB
* **no** = don’t retrieve, just write the next sentence
* **continue** = reuse the chunks retrieved in the previous step

### 2) If `Retrieve = yes`

1. Retrieve **top-K chunks** (normal RAG retrieval): `D = {d1…dK}`
2. For each chunk `di` (batched/parallel):
   
   * Generate **one candidate next sentence** `y_i` using:
     **(prompt + answer-so-far + this chunk)**
    (think of this like different LLMs in parallel, each LLM get one chunk as context so k different outputs (segments)) (whereas in vanilla RAG all chunks go as context in one LLM)

   * Also output critique labels for that candidate:
     * `ISREL`: relevant / irrelevant
     * `ISSUP`: fully supported / partially supported / no support
     * `ISUSE`: 1–5 usefulness
     * `is_final`: whether the answer is complete after this sentence or more generation is needed
3. **Score** all candidate generations (all k distinct segments generated in parallel) and pick the best one
4. Append the **winning sentence(generated segment)** to the answer `y`

✅ Important: we append the **winning sentence (segment)** after every loop so the context for LLM grows after every loop

---

### 3) If `Retrieve = continue`

* Reuse the previous retrieved chunk set `D`
* Generate candidates + critique + pick best (same as above)

---

### 4) If `Retrieve = no`

* Generate the next sentence without retrieval
* Append it to `y`

---

## Candidate scoring (“critic score”)

For each candidate sentence:

* Score ≈ **language-model quality** (how good the sentence is)

  * **weighted critique score**

Critique prefers:

* `ISREL = relevant`
* `ISSUP = fully supported` (partial gets some credit)
* higher `ISUSE`

---

## Optional: segment-level beam search 

Until now we saw picking "1" best segment from k generated segments and appending it after each core loop but paper implements this at a beam search level (Very important that to know how the beam search algorithm works)

Instead of picking only 1 best candidate each step (what we saw earlier-**greedy**), keep **B best partial answers** (**beams**):

* Each beam is a separate draft answer path
* Each step, expand each beam → score → keep top **B**
* At the end, return the best beam


**Beam size B=1 = greedy mode.**

---


