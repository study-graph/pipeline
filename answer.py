"""Answer a question from the indexed material, with checked source citations.

  python answer.py "deine Frage"                 # retrieve, ask the LLM, check citations
  python answer.py "deine Frage" --prompt-only   # print the prompt instead of calling the LLM
  python answer.py "deine Frage" --index other/

The retrieved chunks are numbered [1], [2], ... in the prompt, and the LLM is told to
cite these numbers. Afterwards every cited number is checked against the sources that
were actually given.
"""
import argparse
import sys

import llm
from claims import citations
from config import INDEX_DIR
from store import where

INSTRUCTION = (
    "Beantworte die Frage ausschließlich mit den folgenden Quellen; betone die "
    "Quellen-Basis jedoch nicht am Anfang der Antwort. "
    "Zitiere jede Aussage mit der Nummer der Quelle in eckigen Klammern, z. B. [2]. "
    "Wenn die Quellen die Frage nicht bestimmt beantworten, sage das."
)


def retrieve(question, index):
    from embedder import search                     # imported late: loading torch is slow
    return [chunk for _, chunk in search(question, index)]


def build_prompt(question, chunks):
    """The user part of the prompt: numbered sources, then the question."""
    sources = "\n\n".join(f"[{n}] ({where(c)}: {c['path']})\n{c['text']}"
                          for n, c in enumerate(chunks, 1))
    return f"Quellen:\n{sources}\n\nFrage: {question}\n\nAntwort:"


def check_citations(text, n_sources):
    """-> (cited source numbers that exist, cited numbers that do not), both sorted."""
    cited = citations(text)
    valid = {n for n in cited if 1 <= n <= n_sources}
    return sorted(valid), sorted(cited - valid)


def answer(question, index=INDEX_DIR):
    """Retrieve, generate and check. -> dict with the answer text, the chunks it cites
    as [(number, chunk)], the cited numbers without a source, and all retrieved chunks."""
    chunks = retrieve(question, index)
    text = llm.generate(INSTRUCTION, build_prompt(question, chunks))
    valid, invalid = check_citations(text, len(chunks))
    return {"answer": text, "cited": [(n, chunks[n - 1]) for n in valid],
            "invalid": invalid, "retrieved": chunks}


def add_arguments(ap):
    ap.add_argument("question", nargs="+")
    ap.add_argument("--index", default=INDEX_DIR, help="index directory (default: %(default)s)")
    ap.add_argument("--prompt-only", action="store_true",
                    help="print the prompt instead of calling the LLM")


def run(args):
    """Command line behaviour, shared with `python pipeline.py ask`. -> exit code."""
    question = " ".join(args.question)
    try:
        if args.prompt_only:
            chunks = retrieve(question, args.index)
            print(f"{INSTRUCTION}\n\n{build_prompt(question, chunks)}")
            return 0
        result = answer(question, args.index)
    except (ValueError, llm.LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result["answer"])
    if result["cited"]:
        print("\nQuellen:")
        for n, c in result["cited"]:
            print(f"[{n}] {where(c)}: {c['path']}  ({c['id']})")
    if result["invalid"]:
        print("\nWarnung: zitierte Quellen existieren nicht: "
              + ", ".join(f"[{n}]" for n in result["invalid"]), file=sys.stderr)
    if not result["cited"]:
        print("\nWarnung: die Antwort zitiert keine der Quellen.", file=sys.stderr)
    return 2 if result["invalid"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))
