"""Clean page-wise extracted lecture material into heading-structured markdown.

  python clean.py slides.md                   # -> slides.clean.md
  python clean.py script.md -o out.md --headings keep
  python clean.py deck.md --page-sep '\\n-----\\n' -v

Nothing here is tied to a particular deck. The only assumptions are:
  * pages are separated by a form feed, as parse.py writes them. Input without form
    feeds is split at blank lines only if the resulting blocks show page chrome;
    otherwise it is one page of running text. --page-sep overrides both;
  * page chrome (headers, footers, nav bars, page numbers) is whatever recurs on a
    run of neighbouring pages, while content changes from page to page;
  * a markdown heading on a page, set by parse.py from the font size, is its title.
    Without one, the first line below the header is taken instead.
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


def table_lines(rows, keep):
    """Flatten table rows to plain text unless the extractor's tables are trusted.

    PDF extractors often emit positioned text as pseudo-tables whose cells are single
    words. These cannot be told apart from real tables by their text, so flattening is
    the default: it loses the grid, never the words. Chrome inside a table is then
    found like any other line.
    """
    if keep and len(rows) >= 3 and is_rule(rows[1]):
        return [(r.strip(), None, 0) for r in rows]
    flat = (squash(" ".join(c for c in cells(r) if c)) for r in rows if not is_rule(r))
    return [(t, key_of(t), 0) for t in flat if t]


def load_page(raw, keep_tables=False):
    """A page as a list of (text, key, level). key is None for lines that are never
    chrome and never deduplicated: rows of kept tables and paragraph breaks. level is
    the markdown heading level of the line, 0 for body text."""
    out, table = [], []
    for line in raw.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        if table:
            out += table_lines(table, keep_tables)
            table = []
        marks, text = re.fullmatch(r"\s*(#{1,6}(?=\s))?(.*)", line).groups()
        text = squash(text)
        if text:
            out.append((text, key_of(text), len(marks or "")))
        elif out and out[-1][0]:
            out.append(("", None, 0))               # paragraph break inside a page
    while out and not out[-1][0]:
        out.pop()
    return out


def split_pages(text, page_sep, keep_tables, min_run):
    """-> (pages, how they were separated)."""
    def load(sep):
        return [p for p in (load_page(r, keep_tables) for r in re.split(sep, text)) if p]

    if page_sep:
        return load(page_sep), "--page-sep"
    if "\f" in text:
        return load(r"\f"), "form feed"
    # Some extractors separate pages by blank lines, but so does any markdown text its
    # paragraphs. Blocks are pages only if most of them carry a line that recurs on a
    # run of neighbouring blocks, which paragraphs do not.
    blocks = load(r"\n[ \t]*\n")
    runs = run_sizes([{l[1] for l in p if l[1]} for p in blocks])
    framed = sum(any(l[1] and runs[l[1], i] >= min_run and any(c.isalpha() for c in l[0])
                     for l in p) for i, p in enumerate(blocks))
    if len(blocks) >= min_run and framed > len(blocks) / 2:
        return blocks, "blank lines"
    return load(r"\f"), "none (one page)"


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


def header_length(page, i, runs, min_run, nested, title):
    """Number of leading lines of page i that are header chrome."""
    above = 0
    for j, line in enumerate(page):
        n = runs.get((line[1], i), 0)
        if line is title or n < min_run or n < nested * above:
            return j
        above = n
    return len(page)


def marked_title(page, i, runs, min_run, n_pages):
    """The heading line of a page that changes most often, i.e. has the shortest run.

    A large-font line that stays the same over most of the document is a logo or
    running head, not a title, and is left to the chrome detection.
    """
    marked = [l for l in page if l[2]]
    if not marked:
        return None
    best = min(marked, key=lambda l: (runs.get((l[1], i), 0), l[2]))
    n = runs.get((best[1], i), 0)
    return best if n < min_run or n <= n_pages / 2 else None


def first_line_title(body):
    """Fallback without font information: a short first line containing letters."""
    if body and body[0][1] is not None and len(body[0][0]) <= TITLE_CHARS \
            and any(c.isalpha() for c in body[0][0]):
        return body[0][:2] + (2,)
    return None


def clean(text, page_sep=None, headings="auto", min_run=MIN_RUN, keep_tables=False):
    pages, split = split_pages(text, page_sep, keep_tables, min_run)
    if not pages:
        return "", {"pages": 0, "sections": 0, "headings": "none", "split": split,
                    "chrome": Counter()}

    if headings == "auto":
        words = statistics.median(sum(len(re.findall(r"\w+", l[0])) for l in p) for p in pages)
        headings = "page" if len(pages) > 1 and words <= SLIDE_WORDS else "keep"
    by_page = headings == "page"

    # 1. Header chrome: leading lines that recur on a run of neighbouring pages. On slides
    #    the title is the heading parse.py marked, else the first line below the header,
    #    even if it repeats over a few overlays.
    runs = run_sizes([{l[1] for l in p if l[1]} for p in pages])
    dropped = Counter()
    sections = []                                   # [title or None, [body pages]]
    for i, page in enumerate(pages):
        title = marked_title(page, i, runs, min_run, len(pages)) if by_page else None
        h = header_length(page, i, runs, min_run, NESTED if by_page else 0, title)
        dropped.update(l[0] for l in page[:h])
        body = page[h:]
        if by_page:
            title = title or first_line_title(body)
            body = [l for l in body if not title or l[1] != title[1]]
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
    sec_runs = run_sizes([{l[1] for body in pgs for l in body if l[1]} for _, pgs in sections])
    out = []
    for s, (title, pgs) in enumerate(sections):
        block, seen = [], Counter()
        for body in pgs:
            here = Counter()
            for j, (t, k, level) in enumerate(body):
                edge = j in (0, len(body) - 1)
                if k and sec_runs[k, s] >= min_run and (edge or any(c.isalpha() for c in t)):
                    dropped[t] += 1
                    continue
                here[t] += 1
                if k is None or here[t] > seen[t]:      # skip what an earlier step showed
                    block.append("#" * level + " " + t if level and not by_page else t)
            seen |= here
            block.append("")
        body_text = re.sub(r"\n{3,}", "\n\n", "\n".join(block)).strip()
        head = "#" * title[2] + " " + title[0] if title else ""
        if head or body_text:
            out.append("\n".join(x for x in (head, body_text) if x))

    stats = {"pages": len(pages), "sections": len(out), "headings": headings, "split": split,
             "chrome": dropped}
    return "\n\n".join(out) + "\n", stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="default: <src stem>.clean.md")
    ap.add_argument("--page-sep", help="regex between pages (default: form feed, else detected)")
    ap.add_argument("--headings", choices=["auto", "page", "keep"], default="auto",
                    help="page: one titled section per slide; "
                         "keep: only existing markdown headings (scripts, papers)")
    ap.add_argument("--min-run", type=int, default=MIN_RUN,
                    help="pages a line must recur on to count as chrome (default %(default)s)")
    ap.add_argument("--keep-tables", action="store_true",
                    help="keep markdown tables instead of flattening them to text; "
                         "only for extractors that emit real tables")
    ap.add_argument("-v", "--verbose", action="store_true", help="list the dropped chrome lines")
    args = ap.parse_args()

    text, stats = clean(args.src.read_text(encoding="utf-8"),
                        args.page_sep, args.headings, args.min_run, args.keep_tables)
    out = args.out or args.src.with_name(args.src.stem + ".clean.md")
    out.write_text(text, encoding="utf-8")
    chrome = stats["chrome"]
    print(f"{out}: {stats['pages']} pages (split at: {stats['split']}) -> "
          f"{stats['sections']} sections (headings: {stats['headings']}), {sum(chrome.values())} chrome lines dropped "
          f"({len(chrome)} distinct)")
    if args.verbose:
        for line, n in chrome.most_common():
            print(f"{n:5d}  {line[:100]}")


if __name__ == "__main__":
    main()
