"""File-based store for chunks and their vectors.

An index is a directory with chunks.json, embeddings.npy (row i belongs to chunk i) and
meta.json. The functions here are the only code that knows this layout, so the planned
move to Postgres with pgvector replaces this module and nothing else.
"""
import json
import os
from pathlib import Path

import numpy as np


def load(index):
    """-> (chunks, vectors, meta) of an index; empty values if it does not exist yet."""
    index = Path(index)
    if not (index / "meta.json").exists():
        return [], np.zeros((0, 0), dtype=np.float32), {"model": None, "docs": {}}
    meta = json.loads((index / "meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((index / "chunks.json").read_text(encoding="utf-8"))
    return chunks, np.load(index / "embeddings.npy"), meta


def put(index, doc, chunks, vectors, model, source_hash):
    """Replace everything stored for doc by chunks and their vectors.

    Raises ValueError if the index was built with another embedding model: vectors of
    different models are not comparable, so mixing them would corrupt the search.
    """
    index = Path(index)
    old_chunks, old_vecs, meta = load(index)
    if meta["model"] not in (None, model):
        raise ValueError(f"index {index} uses model {meta['model']}, not {model}; "
                         "rebuild it in a new directory")
    keep = [i for i, c in enumerate(old_chunks) if c["doc"] != doc]
    vectors = np.asarray(vectors, dtype=np.float32).reshape(len(chunks), -1)
    parts = [v for v in (old_vecs[keep], vectors) if len(v)]
    all_vecs = np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)
    all_chunks = [old_chunks[i] for i in keep] + chunks
    meta["model"] = model
    meta["dim"] = int(all_vecs.shape[1]) if len(all_vecs) else None
    meta["docs"][doc] = {"sha256": source_hash, "chunks": len(chunks)}

    # Write to temporary files first, so an interrupted run leaves the old index intact.
    index.mkdir(parents=True, exist_ok=True)
    with open(index / "embeddings.tmp", "wb") as f:
        np.save(f, all_vecs)
    (index / "chunks.tmp").write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (index / "meta.tmp").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    for name in ("embeddings.npy", "chunks.json", "meta.json"):
        os.replace(index / (name.split(".")[0] + ".tmp"), index / name)
