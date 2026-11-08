# Implementation scope

- BM25 retrieves from the supplied local corpus; the bundled data is for smoke evaluation.
- Tuning checks normalized demonstration/development question overlap. Keep a separate test split and check semantic duplicates when adding data.
- Scoring uses normalized exact match; it does not measure evidence recall or reasoning quality.
- Context assembly has a character budget; token-aware packing is a useful extension for larger corpora.
- The Responses API is used through the standard OpenAI SDK. Install the dependencies in `requirements.txt`.
- The retriever is configured per run; concurrent callers should use separate retriever instances.

Run `python -m unittest discover -s tests -v` for offline regression checks.
