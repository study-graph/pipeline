"""Parse a source file into page-wise text, the input format of clean.py.

  python -m ingest.parse slides.pdf                  # -> slides.md
  python -m ingest.parse slides.pdf -o out.md

Pages are separated by a form feed, so later steps can tell them apart without
guessing. The file type is detected from the content, not from the file extension.
In PDFs, lines set in a larger font than the body text become markdown headings.
"""
import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

PAGE_BREAK = "\f"
HEADING_SCALE = 1.15   # a heading is at least this much larger than the body font
SAME_SIZE = 0.05       # font sizes within this relative distance are one heading level
MAX_LEVEL = 4          # deepest heading level the chunker distinguishes
HEADING_CHARS = 200    # longer runs of large text are body text, not a heading


def detect(path):
    """File type by content: 'pdf', 'text', or None if unsupported."""
    with open(path, "rb") as f:
        head = f.read(1024)
    if b"%PDF-" in head:                    # the marker may follow a few junk bytes
        return "pdf"
    if b"\0" in head:
        return None                         # some other binary format
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.end < len(head):               # not merely a character cut off at the end
            return None
    return "text"


def page_lines(page):
    """The lines of a PDF page as [(text, font size)], size being the one most
    characters of the line are set in."""
    lines, buf, sizes = [], [], Counter()

    def end_line():
        size = sizes.most_common(1)[0][0] if sizes else 0.0
        lines.append(("".join(buf).replace(PAGE_BREAK, " ").rstrip(), size))
        buf.clear()
        sizes.clear()

    def visit(text, cm, tm, font, size):
        # The nominal size is scaled by the text matrix and the page transformation.
        scale = math.sqrt(abs((tm[0] * tm[3] - tm[1] * tm[2]) * (cm[0] * cm[3] - cm[1] * cm[2])))
        for n, part in enumerate(text.split("\n")):
            if n:
                end_line()
            buf.append(part)
            if part.strip() and size * scale > 0:
                sizes[round(size * scale, 1)] += len(part.strip())

    page.extract_text(visitor_text=visit)
    end_line()
    return lines


def heading_levels(pages):
    """{font size: heading level} for all sizes clearly above the body font.

    The body font is the size most characters of the document are set in. If every
    line has that size, the result is empty and no heading is marked; clean.py then
    falls back to the position of a line on its page.
    """
    chars = Counter()
    for lines in pages:
        for text, size in lines:
            if size:
                chars[size] += len(text)
    if not chars:
        return {}
    body = chars.most_common(1)[0][0]
    levels, level, anchor = {}, 0, None
    for size in sorted((s for s in chars if s >= HEADING_SCALE * body), reverse=True):
        if anchor is None or size < anchor * (1 - SAME_SIZE):
            level, anchor = min(level + 1, MAX_LEVEL), size
        levels[size] = level
    return levels


def mark_headings(lines, levels):
    """Page text in which consecutive lines of one heading size form one heading."""
    out, i = [], 0
    while i < len(lines):
        level = levels.get(lines[i][1])
        j = i + 1
        while level and j < len(lines) and levels.get(lines[j][1]) == level and lines[j][0].strip():
            j += 1
        title = re.sub(r"\s+", " ", " ".join(t for t, _ in lines[i:j])).strip()
        if level and any(c.isalnum() for c in title) and len(title) <= HEADING_CHARS:
            out.append("#" * level + " " + title)
        else:
            out += [t for t, _ in lines[i:j]]
        i = j
    return "\n".join(out).strip()


def parse_pdf(path):
    """Text of each PDF page. Pages without a text layer (scans) come back empty."""
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(f"{path}: PDF is password-protected")
    pages = [page_lines(page) for page in reader.pages]
    levels = heading_levels(pages)
    return [mark_headings(lines, levels) for lines in pages]


def parse_text(path):
    """Text and markdown files pass through; existing form feeds remain page breaks."""
    return Path(path).read_text(encoding="utf-8").split(PAGE_BREAK)


PARSERS = {"pdf": parse_pdf, "text": parse_text}


def parse(path):
    """-> (pages joined by form feeds, file type, list of pages)."""
    kind = detect(path)
    if kind is None:
        raise ValueError(f"{path}: unsupported file type (supported: {', '.join(PARSERS)})")
    pages = PARSERS[kind](path)
    return PAGE_BREAK.join(pages), kind, pages


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="default: <src stem>.md")
    args = ap.parse_args()

    try:
        text, kind, pages = parse(args.src)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    out = args.out or args.src.with_suffix(".md")
    if out.resolve() == args.src.resolve():
        sys.exit(f"error: output would overwrite the source {args.src}; pass -o")
    out.write_text(text, encoding="utf-8")
    empty = sum(not p for p in pages)
    print(f"{out}: {kind}, {len(pages)} pages"
          + (f", {empty} without text (scanned? needs OCR)" if empty else ""))


if __name__ == "__main__":
    main()
