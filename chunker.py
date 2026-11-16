"""Split cleaned markdown into heading-aware chunks with source metadata.

  python chunker.py slides.clean.md            # -> chunks.json
  python chunker.py slides.clean.md -o out.json --doc netztechnik-0

Each chunk records its document, heading path and page range, and has an id derived
from its content, so an unchanged chunk keeps its id across runs. Sizes are measured
in tokens of the embedding model, so no chunk is truncated when it is embedded.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from config import EMBED_MAX_TOKENS, EMBED_MODEL

MAX_TOKENS = EMBED_MAX_TOKENS - 32   # whole embedded text incl. heading path, with a margin
MIN_TOKENS = 80      # a shorter remainder is appended to the previous chunk of its section
OVERLAP = 50         # tokens repeated from the end of the previous chunk
SMALL_TOKENS = 100   # neighbouring sibling sections are merged up to this combined size
PAGE_MARK = re.compile(r"<!-- page (\d+) -->")


def token_counter(model=EMBED_MODEL):
    """Function text -> number of tokens of the embedding model, without special tokens."""
    from tokenizers import Tokenizer                # imported late: needs the model files
    tokenizer = Tokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def split_sections(md):
    """Yield (heading path as a tuple of titles, lines) per heading section, ignoring
    '#' inside code fences.

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
                yield tuple(h for _, h in path), lines
            level, lines, gap = len(m.group(1)), [], False
            path = [(l, h) for l, h in path if l < level] + [(level, m.group(2).strip())]
        elif raw.strip():
            lines.append((raw.rstrip(), page, gap and bool(lines)))
            gap = False
        else:
            gap = True
    if lines:
        yield tuple(h for _, h in path), lines


def merge_small(sections, count, small):
    """Join consecutive sections under the same parent heading while their combined
    size stays within small tokens.

    A slide with a title and three keywords gives the language model too little context
    on its own. In a merged section each part keeps its title as a line of text, and the
    path names all titles.
    """
    def size(path, lines):
        return count(path[-1] if path else "") + sum(count(t) for t, _, _ in lines)

    groups = []                                     # [parent path, [(path, lines)], tokens]
    for path, lines in sections:
        n = size(path, lines)
        last = groups[-1] if groups else None
        if last and path and last[0] == path[:-1] and last[2] + n <= small:
            last[1].append((path, lines))
            last[2] += n
        else:
            groups.append([path[:-1], [(path, lines)], n])
    for parent, members, _ in groups:
        if len(members) == 1:
            yield members[0]
            continue
        merged = []
        for path, lines in members:
            merged.append((path[-1], lines[0][1], bool(merged)))
            merged += [(t, p, False) if i == 0 else (t, p, g) for i, (t, p, g) in enumerate(lines)]
        yield parent + (" / ".join(path[-1] for path, _ in members),), merged


def split_line(text, limit):
    """A line longer than limit characters as pieces of at most limit characters, cut at
    sentence ends where possible, else between words, else mid-word."""
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


def fit_lines(lines, count, limit):
    """-> (lines, their token counts), with lines above limit tokens split into pieces."""
    out, sizes = [], []
    for text, page, gap in lines:
        n = count(text)
        parts = [text]
        while n > limit and len(parts) < len(text):             # shrink until every piece fits
            chars = max(1, int(len(text) * limit / n * 0.9 / len(parts)))
            parts = split_line(text, chars)
            n = max(count(p) for p in parts) if max(map(len, parts)) > 1 else 0
        for i, part in enumerate(parts):
            out.append((part, page, gap and i == 0))
            sizes.append(count(part))
    return out, sizes


def pack(lines, sizes, max_size, min_size, overlap):
    """Group the lines of one section into chunks of at most max_size tokens.

    A chunk ends at a blank line where that leaves it at least min_size long, otherwise
    between two lines. Each following chunk starts with the last lines of the previous
    one, up to overlap tokens, so a statement cut at the boundary stays retrievable.
    """
    # Size of the block (paragraph) starting at each line.
    block = [0] * len(lines)
    for i in range(len(lines) - 1, -1, -1):
        rest = block[i + 1] if i + 1 < len(lines) and not lines[i + 1][2] else 0
        block[i] = sizes[i] + rest

    chunks, cur, size, fresh = [], [], 0, 0         # cur: line indices; fresh: not overlap
    for i, line in enumerate(lines):
        n = sizes[i]
        at_block = line[2] and size >= min_size and block[i] <= max_size < size + block[i]
        if fresh and (size + n > max_size or at_block):
            chunks.append(cur)
            tail, kept = [], 0
            for j in reversed(cur):
                if kept + sizes[j] > overlap:
                    break
                tail.insert(0, j)
                kept += sizes[j]
            cur, size, fresh = tail, kept, 0
        if not fresh and size + n > max_size:       # the overlap would crowd out new text
            cur, size = [], 0
        cur.append(i)
        size, fresh = size + n, fresh + 1
    new = cur[len(cur) - fresh:]
    new_size = sum(sizes[j] for j in new)
    if chunks and new_size < min_size and sum(sizes[j] for j in chunks[-1]) + new_size <= max_size:
        chunks[-1] = chunks[-1] + new               # no tiny chunk of leftovers
    elif fresh:
        chunks.append(cur)
    return [[lines[j] for j in part] for part in chunks]


def chunk(md, doc, count=None, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS,
          overlap=OVERLAP, small_tokens=SMALL_TOKENS):
    """Chunks of a cleaned markdown document as a list of dicts.

    doc identifies the source document. It is stored with every chunk and enters the
    chunk id, so equal text in two documents yields two chunks. count maps a text to
    its size; by default it counts tokens of the embedding model.
    """
    count = count or token_counter()
    chunks, seen = [], set()
    for titles, lines in merge_small(split_sections(md), count, small_tokens):
        path = " > ".join(titles)
        prefix = f"passage: {path}\n"               # e5 prefix + heading context
        # Room for the text itself; the 8 covers special tokens and joining lines.
        room = max(max_tokens - count(prefix) - 8, max_tokens // 4)
        lines, sizes = fit_lines(lines, count, room // 2)
        for part in pack(lines, sizes, room, min_tokens, overlap):
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
                "embed_text": prefix + text,
            })
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("chunks.json"))
    ap.add_argument("--doc", help="document id (default: file name without .clean.md)")
    args = ap.parse_args()

    doc = args.doc or args.src.name.removesuffix(".md").removesuffix(".clean")
    count = token_counter()
    chunks = chunk(args.src.read_text(encoding="utf-8"), doc, count)
    args.out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = sorted(count(c["embed_text"]) + 2 for c in chunks) or [0]
    print(f"{args.out}: {len(chunks)} chunks for '{doc}', "
          f"{sizes[0]}-{sizes[-1]} tokens (median {sizes[len(sizes) // 2]})")


if __name__ == "__main__":
    main()
