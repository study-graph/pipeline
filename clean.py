"""Clean page-wise extracted lecture material into heading-structured markdown.

  python clean.py netze.md                    # -> netze.clean.md
  python clean.py script.md -o out.md --headings keep
  python clean.py deck.md --page-sep '\\n-----\\n' -v

Nothing here is tied to a particular deck. The only assumptions are:
  * the input is split into pages (form feed, else blank lines, else --page-sep);
  * page chrome (headers, footers, nav bars, page numbers) is whatever recurs on a
    run of neighbouring pages, while content changes from page to page.
"""
import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MIN_RUN = 4        # a line must recur on this many neighbouring pages/sections to be chrome
MAX_GAP = 3        # occurrences at most this many pages apart still belong to one run
NESTED = 0.5       # a top line whose run is shorter than this share of the line above it
                   # is a title repeated over a few pages, not a second header line
TABLE_FILL = 0.7   # share of non-empty cells above which a table is real, not layout
SLIDE_WORDS = 120  # median words per page up to which the input is treated as slides
TITLE_CHARS = 120  # longer first lines are body text, not slide titles


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


def key_of(text):
    """Comparison key: digits are masked so 'Seite 3' and 'Seite 4' (or '4/25') match."""
    return re.sub(r"\d+", "#", text).casefold()


def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def is_rule(row):
    cs = cells(row)
    return any(cs) and all(re.fullmatch(r":?-+:?|", c) for c in cs)


def table_lines(rows):
    """Keep a densely filled table as markdown; flatten one that only encodes layout.

    PDF extractors often emit positioned text as a sparse pseudo-table. Such rows carry
    no tabular meaning, so their cells are joined into plain text.
    """
    body = [cells(r) for r in rows if not is_rule(r)]
    filled = sum(bool(c) for r in body for c in r) / max(1, sum(len(r) for r in body))
    if len(rows) >= 3 and is_rule(rows[1]) and filled >= TABLE_FILL:
        return [(r.strip(), None) for r in rows]
    flat = (squash(" ".join(c for c in r if c)) for r in body)
    return [(t, key_of(t)) for t in flat if t]


def load_page(raw):
    """A page as a list of (text, key). key is None for lines that are never chrome
    and never deduplicated: rows of real tables and paragraph breaks."""
    out, table = [], []
    for line in raw.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        if table:
            out += table_lines(table)
            table = []
        text = squash(line)
        if text:
            out.append((text, key_of(text)))
        elif out and out[-1][0]:
            out.append(("", None))                  # paragraph break inside a page
    while out and not out[-1][0]:
        out.pop()
    return out


def run_sizes(units):
    """units: one set of keys per page/section -> {(key, index): size of its run}."""
    where = defaultdict(list)
    for i, keys in enumerate(units):
        for k in keys:
            where[k].append(i)
    size = {}
    for k, idx in where.items():
        run = [idx[0]]
        for i in idx[1:] + [None]:
            if i is not None and i - run[-1] <= MAX_GAP:
                run.append(i)
                continue
            for j in run:
                size[k, j] = len(run)
            run = [i]
    return size


def header_length(page, i, runs, min_run, nested):
    """Number of leading lines of page i that are header chrome."""
    above = 0
    for j, (_, k) in enumerate(page):
        n = runs.get((k, i), 0)
        if n < min_run or n < nested * above:
            return j
        above = n
    return len(page)


def is_title(text, key):
    return key is not None and len(text) <= TITLE_CHARS and any(c.isalpha() for c in text)


def clean(text, page_sep=None, headings="auto", min_run=MIN_RUN):
    sep = page_sep or (r"\f" if "\f" in text else r"\n[ \t]*\n")
    pages = [p for p in map(load_page, re.split(sep, text)) if p]
    if not pages:
        return "", {"pages": 0, "sections": 0, "headings": "none", "chrome": Counter()}

    if headings == "auto":
        words = statistics.median(sum(len(t.split()) for t, _ in p) for p in pages)
        has_md = sum(any(t.startswith("#") for t, _ in p) for p in pages) / len(pages)
        headings = "page" if words <= SLIDE_WORDS and has_md < 0.2 else "keep"
    by_page = headings == "page"

    # 1. Header chrome: leading lines that recur on a run of neighbouring pages. On slides
    #    the first line below them is the title, even if it repeats over a few overlays.
    runs = run_sizes([{k for _, k in p if k} for p in pages])
    dropped = Counter()
    sections = []                                   # [title or None, [body pages]]
    for i, page in enumerate(pages):
        h = header_length(page, i, runs, min_run, NESTED if by_page else 0)
        dropped.update(t for t, _ in page[:h])
        body, title = page[h:], None
        if by_page and body and is_title(*body[0]):
            title, body = body[0], [l for l in body[1:] if l[1] != body[0][1]]
        # Consecutive pages with the same title are one slide shown in several steps.
        same = sections and title and sections[-1][0] and sections[-1][0][1] == title[1]
        if same or (by_page and sections and not title):
            sections[-1][1].append(body)
        else:
            sections.append([title, [body]])

    # 2. Remaining chrome (footers, displaced headers): lines recurring across a run of
    #    sections. Counting sections rather than pages keeps content that is merely
    #    repeated by the overlays of one slide. Lines without letters only count at the
    #    page edge, where page numbers sit; elsewhere they are data.
    sec_runs = run_sizes([{k for body in pgs for _, k in body if k} for _, pgs in sections])
    out = []
    for s, (title, pgs) in enumerate(sections):
        block, seen = [], Counter()
        for body in pgs:
            here = Counter()
            for j, (t, k) in enumerate(body):
                edge = j in (0, len(body) - 1)
                if k and sec_runs[k, s] >= min_run and (edge or any(c.isalpha() for c in t)):
                    dropped[t] += 1
                    continue
                here[t] += 1
                if k is None or here[t] > seen[t]:      # skip what an earlier step showed
                    block.append(t)
            seen |= here
            block.append("")
        body_text = re.sub(r"\n{3,}", "\n\n", "\n".join(block)).strip()
        head = f"## {title[0].lstrip('# ')}" if title else ""
        if head or body_text:
            out.append("\n".join(x for x in (head, body_text) if x))

    stats = {"pages": len(pages), "sections": len(out), "headings": headings, "chrome": dropped}
    return "\n\n".join(out) + "\n", stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="default: <src stem>.clean.md")
    ap.add_argument("--page-sep", help="regex between pages (default: form feed, else blank line)")
    ap.add_argument("--headings", choices=["auto", "page", "keep"], default="auto",
                    help="page: first line of each page becomes a heading (slides); "
                         "keep: only existing markdown headings (scripts, papers)")
    ap.add_argument("--min-run", type=int, default=MIN_RUN,
                    help="pages a line must recur on to count as chrome (default %(default)s)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list the dropped chrome lines")
    args = ap.parse_args()

    text, stats = clean(args.src.read_text(encoding="utf-8"),
                        args.page_sep, args.headings, args.min_run)
    out = args.out or args.src.with_name(args.src.stem + ".clean.md")
    out.write_text(text, encoding="utf-8")
    chrome = stats["chrome"]
    print(f"{out}: {stats['pages']} pages -> {stats['sections']} sections "
          f"(headings: {stats['headings']}), {sum(chrome.values())} chrome lines dropped "
          f"({len(chrome)} distinct)")
    if args.verbose:
        for line, n in chrome.most_common():
            print(f"{n:5d}  {line[:100]}")


if __name__ == "__main__":
    main()
