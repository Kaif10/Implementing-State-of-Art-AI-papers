# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

An inspired implementation of [Self-RAG](https://arxiv.org/abs/2310.11511) using off-the-shelf LLM APIs.

## Paper methodology (in plain English)

The inference loop combines retrieval with segment-level self-critique:

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
2. For each chunk `di` (evaluated sequentially):
   
   * Generate **one candidate next sentence** `y_i` using:
     **(prompt + answer-so-far + this chunk)**
    Each call sees one passage; ordinary RAG typically places several passages in a single prompt.

   * Also output critique labels for that candidate:
     * `ISREL`: relevant / irrelevant
     * `ISSUP`: fully supported / partially supported / no support
     * `ISUSE`: 1–5 usefulness
     * `is_final`: whether the answer is complete after this sentence or more generation is needed
3. **Score** all candidate generations (the candidate segments) and pick the best one
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

Each candidate receives a weighted sum of discrete relevance, support, and
usefulness labels. These are prompted judgments, not trained reflection-token
probabilities or a language-model likelihood score.

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



## Run

From this folder, using Python 3.12:

```bash
pip install -r requirements.txt
python main.py ingest --docs kb --out faiss_index
python main.py ask --index faiss_index "What are the security requirements?"
python -m unittest discover -s tests -v
```

Set `OPENAI_API_KEY` before ingestion or answering. Offline tests need only
`openai` and `pydantic`; they use synthetic passages and make no API calls.

This adaptation uses API-generated critique labels and sequential passage calls.
Beam ranking uses mean segment score to avoid automatically rewarding longer
answers. If every passage is rejected, both decoding modes request a no-context
fallback. Rebuild the index when the documents or embedding settings change.
