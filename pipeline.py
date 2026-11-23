"""Entry point of the pipeline.

  python pipeline.py ingest slides.pdf script.md      # parse -> clean -> chunk -> embed -> store
  python pipeline.py ingest corpus/ --index index     # every supported file below corpus/
  python pipeline.py ingest slides.pdf --save-steps debug/
  python pipeline.py ask "Wie funktioniert der 3-way Handshake?"

Every step is a function call; nothing is passed between steps through files. A file
whose content has not changed since its last ingest is skipped, and of a changed file
only the chunks with new content are embedded again.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from answering import answer
from config import EMBED_MODEL, INDEX_DIR
from ingest import chunker, clean, parse
from retrieval import store


def ingest(path, index=INDEX_DIR, count=None, force=False, save_steps=None):
    """Run one source file through all steps. -> dict describing what was done."""
    path = Path(path)
    doc = path.stem
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    old_chunks, old_vecs, meta = store.load(index)
    if not force and meta["docs"].get(doc, {}).get("sha256") == source_hash:
        return {"doc": doc, "status": "unchanged"}

    text, kind, pages = parse.parse(path)
    cleaned, stats = clean.clean(text)
    chunks = chunker.chunk(cleaned, doc, count)
    if save_steps:
        save_steps = Path(save_steps)
        save_steps.mkdir(parents=True, exist_ok=True)
        (save_steps / f"{doc}.md").write_text(text, encoding="utf-8")
        (save_steps / f"{doc}.clean.md").write_text(cleaned, encoding="utf-8")
    if not chunks:
        raise ValueError(f"{path}: no text found" + (" (scanned PDF? needs OCR)"
                                                     if kind == "pdf" else ""))

    # Chunk ids are content hashes: a vector stored under the same id is still valid.
    known = {c["id"]: old_vecs[i] for i, c in enumerate(old_chunks)} \
        if meta["model"] == EMBED_MODEL else {}
    new = [c for c in chunks if c["id"] not in known]
    if new:
        from retrieval import embedder                             # imported late: loading torch is slow
        known.update(zip((c["id"] for c in new), embedder.encode([c["embed_text"] for c in new])))
    store.put(index, doc, chunks, [known[c["id"]] for c in chunks], EMBED_MODEL, source_hash)
    return {"doc": doc, "status": "ingested", "kind": kind, "pages": stats["pages"],
            "sections": stats["sections"], "chunks": len(chunks), "embedded": len(new)}


def source_files(paths):
    """The given files, plus every file with a supported type below given directories."""
    for path in map(Path, paths):
        if path.is_dir():
            yield from (p for p in sorted(path.rglob("*"))
                        if p.is_file() and not p.name.startswith(".") and parse.detect(p))
        else:
            yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="add source files to the index")
    ing.add_argument("paths", nargs="+", help="files or directories")
    ing.add_argument("--index", default=INDEX_DIR, help="index directory (default: %(default)s)")
    ing.add_argument("--force", action="store_true", help="also process unchanged files")
    ing.add_argument("--save-steps", metavar="DIR",
                     help="also write the parsed and the cleaned text there, for inspection")
    answer.add_arguments(sub.add_parser("ask", help="answer a question from the index"))
    args = ap.parse_args()
    if args.command == "ask":
        sys.exit(answer.run(args))

    count, failed = chunker.token_counter(), 0
    for path in source_files(args.paths):
        try:
            r = ingest(path, args.index, count, args.force, args.save_steps)
        except (OSError, ValueError) as e:
            failed += 1
            print(f"error: {e}" if str(path) in str(e) else f"{path}: error: {e}", file=sys.stderr)
            continue
        if r["status"] == "unchanged":
            print(f"{path}: unchanged, skipped")
        else:
            print(f"{path}: {r['kind']}, {r['pages']} pages -> {r['sections']} sections -> "
                  f"{r['chunks']} chunks, {r['embedded']} embedded")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
