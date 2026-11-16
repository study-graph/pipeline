"""Split cleaned markdown into heading-aware chunks with source metadata.

  python chunker.py slides.clean.md            # -> chunks.json
  python chunker.py slides.clean.md -o out.json --doc netztechnik-0

Each chunk records its document, heading path and page range, and has an id derived
from its content, so an unchanged chunk keeps its id across runs.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

MAX_CHARS = 1800   # about 450 tokens
MIN_CHARS = 300    # a shorter remainder is appended to the previous chunk of its section
OVERLAP = 200      # characters repeated from the end of the previous chunk
PAGE_MARK = re.compile(r"<!-- page (\d+) -->")


def split_sections(md):
    """Yield (heading path, lines) per heading section, ignoring '#' inside code fences.

    A line is (text, page, starts_block): page comes from the page marks written by
    clean.py (None without them), starts_block is set after a blank line.
    """
    path, lines, in_code, page, gap = [], [], False, None, False
    for raw in md.splitlines():
        mark = PAGE_MARK.fullmatch(raw.strip())
        if mark:
            page = int(mark.group(1))
            continue
        if raw.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", raw)
        if m:
            if lines:
                yield " > ".join(h for _, h in path), lines
            level, lines, gap = len(m.group(1)), [], False
            path = [(l, h) for l, h in path if l < level] + [(level, m.group(2).strip())]
        elif raw.strip():
            for part in split_line(raw.rstrip(), MAX_CHARS // 2):
                lines.append((part, page, gap and bool(lines)))
                gap = False
        else:
            gap = True
    if lines:
        yield " > ".join(h for _, h in path), lines


def split_line(text, limit):
    """A line longer than limit as pieces of at most limit characters, cut at sentence
    ends where possible, else between words, else mid-word."""
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?;:])\s+", text):
        for word in sentence.split() if len(sentence) > limit else [sentence]:
            while len(word) > limit:
                out, cur, word = out + ([cur] if cur else []) + [word[:limit]], "", word[limit:]
            if cur and len(cur) + 1 + len(word) > limit:
                out, cur = out + [cur], ""
            cur = f"{cur} {word}" if cur else word
    return out + ([cur] if cur else [])


def pack(lines, max_chars, min_chars, overlap):
    """Group the lines of one section into chunks of at most max_chars characters.

    A chunk ends at a blank line where that leaves it at least min_chars long, otherwise
    between two lines. Each following chunk starts with the last lines of the previous
    one, up to overlap characters, so a statement cut at the boundary stays retrievable.
    """
    # Length of the block (paragraph) starting at each line.
    block = [0] * len(lines)
    for i in range(len(lines) - 1, -1, -1):
        rest = block[i + 1] if i + 1 < len(lines) and not lines[i + 1][2] else 0
        block[i] = len(lines[i][0]) + 1 + rest

    chunks, cur, size, fresh = [], [], 0, 0         # fresh: lines in cur that are not overlap
    for i, line in enumerate(lines):
        n = len(line[0]) + 1
        at_block = line[2] and size >= min_chars and block[i] <= max_chars < size + block[i]
        if fresh and (size + n > max_chars or at_block):
            chunks.append(cur)
            tail, kept = [], 0
            for prev in reversed(cur):
                if kept + len(prev[0]) + 1 > overlap:
                    break
                tail.insert(0, prev)
                kept += len(prev[0]) + 1
            cur, size, fresh = tail, kept, 0
        if not fresh and size + n > max_chars:      # the overlap would crowd out new text
            cur, size = [], 0
        cur.append(line)
        size, fresh = size + n, fresh + 1
    new = cur[len(cur) - fresh:]
    if chunks and sum(len(t) + 1 for t, _, _ in new) < min_chars:
        chunks[-1] = chunks[-1] + new               # no tiny chunk of leftovers
    elif fresh:
        chunks.append(cur)
    return chunks


def chunk(md, doc, max_chars=MAX_CHARS, min_chars=MIN_CHARS, overlap=OVERLAP):
    """Chunks of a cleaned markdown document as a list of dicts.

    doc identifies the source document. It is stored with every chunk and enters the
    chunk id, so equal text in two documents yields two chunks.
    """
    chunks, seen = [], set()
    for path, lines in split_sections(md):
        for part in pack(lines, max_chars, min_chars, overlap):
            text = "".join(("\n\n" if gap and i else "\n" if i else "") + t
                           for i, (t, _, gap) in enumerate(part))
            cid = hashlib.sha256(f"{doc}\n{path}\n{text}".encode()).hexdigest()[:16]
            if cid in seen:                         # identical repeat within the document
                continue
            seen.add(cid)
            pages = [p for _, p, _ in part if p is not None]
            chunks.append({
                "id": cid,
                "doc": doc,
                "path": path,
                "pages": [min(pages), max(pages)] if pages else None,
                "text": text,
                "embed_text": f"passage: {path}\n{text}",   # e5 prefix + heading context
            })
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("chunks.json"))
    ap.add_argument("--doc", help="document id (default: file name without .clean.md)")
    args = ap.parse_args()

    doc = args.doc or args.src.name.removesuffix(".md").removesuffix(".clean")
    chunks = chunk(args.src.read_text(encoding="utf-8"), doc)
    args.out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = sorted(len(c["text"]) for c in chunks) or [0]
    print(f"{args.out}: {len(chunks)} chunks for '{doc}', "
          f"{sizes[0]}-{sizes[-1]} characters (median {sizes[len(sizes) // 2]})")


if __name__ == "__main__":
    main()
