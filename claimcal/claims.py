"""Split an answer into claims, each with the source numbers that back it.

A claim is a single factual statement that can be checked on its own. Each claim keeps the
numbers of the sources the answer cites for the statement it comes from, so verify.py can
check it against exactly these sources. A claim from a statement without citation gets no
sources.

Two methods produce the same output, a list of {"text": ..., "sources": [n, ...]}:

  llm        The verify model splits the answer into atomic claims and resolves
             references, as in CLAIM-CAL (Pal 2026). One model call per answer.
  sentences  Every sentence or list item is one claim. No model call; a baseline that is
             coarser, since a sentence can hold several claims.

  python -m claimcal.claims answer.txt   # JSON list of claims on stdout
                                                   # (a source list as printed by
                                                   # `pipeline.py ask` is ignored)
  python -m claimcal.claims - --method sentences < answer.txt
  python -m claimcal.claims answer.txt --question "Wie funktioniert der 3-way Handshake?"
"""
import argparse
import json
import re
import sys

import llm

METHODS = ("llm", "sentences")
CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

INSTRUCTION = (
    "Du zerlegst eine Antwort in atomare Aussagen (Claims). Ein Claim ist eine einzelne "
    "Tatsachenaussage, die ohne die übrige Antwort verständlich und prüfbar ist: Löse "
    "Pronomen und Bezüge auf und nenne den Gegenstand ausdrücklich. Das gilt auch für "
    "Ausdrücke, die nur im Zusammenhang verständlich sind, etwa \"das Paket\" oder \"im "
    "dritten Schritt\": Ergänze, worauf sie sich beziehen. Übernimm nur, was in der "
    "Antwort steht; ergänze, korrigiere und bewerte nichts. Lass Sätze ohne Tatsachenaussage "
    "weg, etwa Überleitungen oder den Hinweis, dass die Quellen eine Frage nicht beantworten. "
    "Gib zu jedem Claim die Quellennummern an, mit denen die Antwort die Aussage belegt, aus "
    "der er stammt, und eine leere Liste, wenn diese Aussage kein Zitat hat. Der Claim-Text "
    "selbst enthält keine Quellennummern. "
    'Format: {"claims": [{"text": "...", "sources": [1, 2]}]}'
)


def citations(text):
    """-> the source numbers cited in text, as a set."""
    return {int(n) for group in CITATION.findall(text) for n in re.findall(r"\d+", group)}


def strip_citations(text):
    """-> text without citation markers and without the space left before them."""
    return re.sub(r"\s+([.,;:!?])", r"\1", CITATION.sub("", text)).strip()


def decompose(answer, question=None, method="llm"):
    """-> claims of answer, a list of {"text", "sources"} in the order of the answer."""
    if method == "llm":
        return _by_llm(answer, question)
    if method == "sentences":
        return _by_sentences(answer)
    raise ValueError(f"unknown method {method!r} (expected one of {METHODS})")


def _by_llm(answer, question):
    prompt = (f"Frage: {question}\n\n" if question else "") + f"Antwort:\n{answer}"
    data = llm.generate_json(INSTRUCTION, prompt, role="verify")
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise llm.LLMError(f'decomposition is not of the form {{"claims": [...]}}: '
                           f"{str(data)[:200]}")
    found = validate(data["claims"], citations(answer))
    # An answer that cites sources states something: an empty result is a failed
    # extraction, not an answer without claims (fallback as in Pal 2026).
    return found or (_by_sentences(answer) if citations(answer) else [])


def validate(raw_claims, cited):
    """-> the model's claims in normal form, keeping only sources the answer really cites.

    A source number the model assigns but the answer does not cite is dropped: a claim
    must not gain evidence through the decomposition. Items without text are skipped.
    """
    claims = []
    for item in raw_claims:
        if not isinstance(item, dict):
            raise llm.LLMError(f"claim is not an object: {str(item)[:100]}")
        text = strip_citations(str(item.get("text") or ""))
        if not text:
            continue
        sources = item.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]
        numbers = set()
        for s in sources:
            try:
                numbers.add(int(s))
            except (TypeError, ValueError):
                pass
        claims.append({"text": text, "sources": sorted(numbers & cited)})
    return claims


# Abbreviations whose period does not end a sentence, compared in lower case.
ABBREVIATIONS = {"z.", "b.", "bzw.", "d.", "h.", "u.", "a.", "s.", "vgl.", "ca.", "evtl.",
                 "ggf.", "inkl.", "usw.", "etc.", "nr.", "abb.", "kap.", "bspw.", "e.g.",
                 "i.e.", "vs.", "dr.", "prof.", "max.", "min."}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
LEADING_CITATIONS = re.compile(r"^((?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])+)\s*")
# The list of cited sources that `pipeline.py ask` prints below the answer.
SOURCE_LIST = re.compile(r"\n\s*Quellen:\s*\n(?:\s*\[\d+\][^\n]*(?:\n|$))+\s*$")


def _split_sentences(line):
    """-> the sentences of one line; ends at '.', '!' or '?' unless after an abbreviation
    or a number ("3. Schicht")."""
    parts, start = [], 0
    for m in SENTENCE_END.finditer(line):
        before = line[start:m.start()].split()
        last = before[-1].lower() if before else ""
        if last in ABBREVIATIONS or re.fullmatch(r"\d+\.", last):
            continue
        parts.append(line[start:m.start()])
        start = m.end()
    parts.append(line[start:])
    return [p for p in (p.strip() for p in parts) if p]


def _by_sentences(answer):
    units = []
    for line in answer.splitlines():
        line = LIST_MARKER.sub("", line).strip()
        if line.startswith("#"):                    # a markdown heading states nothing
            continue
        sentences = _split_sentences(line)
        # A citation placed after the period ("... Handshake. [2]") belongs to the
        # sentence before it. Only within a line: a line starting with a citation is
        # not a continuation of the line above.
        for i in range(1, len(sentences)):
            m = LEADING_CITATIONS.match(sentences[i])
            if m:
                sentences[i - 1] += " " + m.group(1).strip()
                sentences[i] = sentences[i][m.end():]
        units.extend(sentences)

    claims = []
    for unit in units:
        text = strip_citations(unit)
        if re.search(r"\w", text):
            claims.append({"text": text, "sources": sorted(citations(unit))})
    return claims


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("answer", help="file with the answer text, or - for stdin")
    ap.add_argument("--question", help="the question, to resolve references in the answer")
    ap.add_argument("--method", choices=METHODS, default="llm")
    args = ap.parse_args()
    try:
        text = sys.stdin.read() if args.answer == "-" else open(args.answer,
                                                                encoding="utf-8").read()
        claims = decompose(SOURCE_LIST.sub("", text), args.question, args.method)
    except (OSError, ValueError, llm.LLMError) as e:
        raise SystemExit(f"error: {e}")
    print(json.dumps(claims, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
