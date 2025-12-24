from __future__ import annotations

import glob
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI


# Data

@dataclass(frozen=True)
class Chunk:
    vec_id: int
    chunk_id: str   # e.g. "security_policy.md:0"
    source: str     # full path
    text: str


@dataclass(frozen=True)
class Meta:
    embed_model: str
    embed_dimensions: Optional[int]
    chunks: List[Chunk]



# Docs + chunking

def chunk_text(text: str, size_chars: int = 3000, overlap_chars: int = 300) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: List[str] = []
    i = 0
    while i < len(text):
        j = min(i + size_chars, len(text))
        out.append(text[i:j].strip())
        if j == len(text):
            break
        i = max(0, j - overlap_chars)
    return [c for c in out if c]


def read_file(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # optional dependency
        r = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in r.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def list_docs(folder: str) -> List[str]:
    pats = ("*.txt", "*.md", "*.markdown", "*.pdf")
    files: List[str] = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    return sorted(set(files))



# Embeddings + FAISS

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n


class Embedder:
    def __init__(self, client: OpenAI, model: str, dimensions: Optional[int]):
        self.client = client
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs: List[List[float]] = []
        bs = 64
        for i in range(0, len(texts), bs):
            batch = [t.replace("\n", " ") for t in texts[i:i + bs]]
            kwargs = {"model": self.model, "input": batch, "encoding_format": "float"}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)
            vecs.extend([d.embedding for d in resp.data])
        return np.asarray(vecs, dtype=np.float32)


def _build_faiss(vectors: np.ndarray, ids: np.ndarray) -> faiss.Index:
    # cosine similarity via inner product on L2-normalized vectors
    vectors = _l2_normalize(vectors.astype(np.float32))
    dim = vectors.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, ids.astype(np.int64))
    return index


def _faiss_search(index: faiss.Index, q: np.ndarray, top_k: int) -> List[int]:
    q = _l2_normalize(q.astype(np.float32))
    _, ids = index.search(q, top_k)
    return [int(i) for i in ids[0] if int(i) != -1]



# KB wrapper
class FaissKB:
    def __init__(self, index: faiss.Index, meta: Meta, embedder: Embedder):
        self.index = index
        self.meta = meta
        self.embedder = embedder
        self.by_vec_id = {c.vec_id: c for c in meta.chunks}

    @staticmethod
    def build(
        client: OpenAI,
        docs_dir: str,
        out_dir: str,
        *,
        embed_model: str = "text-embedding-3-large",
        embed_dimensions: Optional[int] = 1024,
        chunk_chars: int = 3000,
        overlap_chars: int = 300,
    ) -> None:
        files = list_docs(docs_dir)
        if not files:
            raise SystemExit(f"No documents found under {docs_dir}")

        chunks: List[Chunk] = []
        vid = 0
        for fp in files:
            txt = read_file(fp).replace("\x00", " ").strip()
            if not txt:
                continue
            for i, ch in enumerate(chunk_text(txt, chunk_chars, overlap_chars)):
                chunks.append(Chunk(vec_id=vid, chunk_id=f"{Path(fp).name}:{i}", source=fp, text=ch))
                vid += 1

        embedder = Embedder(client, embed_model, embed_dimensions)

        vecs: List[np.ndarray] = []
        bs = 64
        for i in range(0, len(chunks), bs):
            batch = [c.text for c in chunks[i:i + bs]]
            vecs.append(embedder.embed(batch))
            print(f"embedded {min(i+bs, len(chunks))}/{len(chunks)}", end="\r")
        print()

        mat = np.vstack(vecs).astype(np.float32)
        ids = np.array([c.vec_id for c in chunks], dtype=np.int64)
        index = _build_faiss(mat, ids)

        meta = Meta(embed_model=embed_model, embed_dimensions=embed_dimensions, chunks=chunks)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(out / "vectors.faiss"))
        with open(out / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved index -> {out_dir}")
        print(f"chunks={len(chunks)} dim={index.d}")

    @staticmethod
    def load(client: OpenAI, index_dir: str) -> "FaissKB":
        p = Path(index_dir)
        index = faiss.read_index(str(p / "vectors.faiss"))
        with open(p / "meta.pkl", "rb") as f:
            meta: Meta = pickle.load(f)
        embedder = Embedder(client, meta.embed_model, meta.embed_dimensions)
        return FaissKB(index=index, meta=meta, embedder=embedder)

    def search(self, query: str, top_k: int) -> List[Chunk]:
        qv = self.embedder.embed([query])
        ids = _faiss_search(self.index, qv, top_k)
        return [self.by_vec_id[i] for i in ids if i in self.by_vec_id]
