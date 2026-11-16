"""Embed texts with the local model, or search an index from the command line.

  python embedder.py "deine Frage"              # top-k chunks of the default index
  python embedder.py "deine Frage" --index other/

The index is built by `python pipeline.py ingest`.
"""
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

import store
from config import EMBED_MODEL as MODEL, INDEX_DIR

TOP_K = 5

_model = None
def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model


def encode(texts):
    # e5 is trained for cosine similarity -> L2-normalise so dot product == cosine.
    return model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def search(question, index=INDEX_DIR, k=TOP_K):
    """-> [(score, chunk)] of the k chunks most similar to the question."""
    chunks, vecs, meta = store.load(index)
    if not chunks:
        raise ValueError(f"index {index} is empty; run `python pipeline.py ingest` first")
    if meta["model"] != MODEL:
        raise ValueError(f"index {index} uses model {meta['model']}, not {MODEL}")
    q = encode([f"query: {question}"])[0]          # e5 wants the "query:" prefix
    scores = vecs @ q                              # cosine, since both are normalised
    return [(float(scores[i]), chunks[i]) for i in np.argsort(-scores)[:k]]


def where(chunk):
    """Human-readable source of a chunk, e.g. 'netze, S. 3-4'."""
    pages = chunk["pages"]
    if not pages:
        return chunk["doc"]
    return f"{chunk['doc']}, S. {pages[0]}" + (f"-{pages[1]}" if pages[1] != pages[0] else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="+")
    ap.add_argument("--index", default=INDEX_DIR)
    args = ap.parse_args()
    try:
        hits = search(" ".join(args.question), args.index)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    for score, c in hits:
        print(f"\n[{c['id']}] {score:.3f}  {where(c)}  {c['path']}")
        print(c["text"][:300].strip())
