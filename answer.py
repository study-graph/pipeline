"""Assemble a citation-ready RAG prompt for a question.

  python3 answer.py "deine Frage"
  python3 answer.py "deine Frage" --index other/

Retrieves the top-k chunks and prints a prompt you can paste into any LLM
(Gemini Flash, etc.). The LLM is told to answer only from the chunks and cite [id]s.
"""
import argparse
import sys

from config import INDEX_DIR
from embedder import search, where

INSTRUCTION = (
    "Beantworte die Frage ausschließlich mit den folgenden Quellen; betone die Quellen-Basis jedoch nicht am Anfang der Antwort."
    "Zitiere jede Aussage mit der Chunk-Nummer in eckigen Klammern, z. B. [25]. "
    "Wenn die Quellen die Frage nicht bestimmt beantworten, sage das.\n"
)


def build_prompt(question, index=INDEX_DIR):
    sources = "\n\n".join(
        f"[{c['id']}] ({where(c)}: {c['path']})\n{c['text']}" for _, c in search(question, index)
    )
    return f"{INSTRUCTION}\nQuellen:\n{sources}\n\nFrage: {question}\n\nAntwort:"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="+")
    ap.add_argument("--index", default=INDEX_DIR)
    args = ap.parse_args()
    try:
        print(build_prompt(" ".join(args.question), args.index))
    except ValueError as e:
        sys.exit(f"error: {e}")
