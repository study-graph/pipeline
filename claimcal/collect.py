"""Collect answers with raw confidence and label them, as input for calibrate.py.

  python -m claimcal.collect run data/questions.txt   # one question per line, # = comment
  python -m claimcal.collect label data/runs/runs.json   # judge each answer: r, f, s or q
  python -m claimcal.calibrate fit data/runs/runs.json   # fit on the labelled answers

Both steps save after every answer and continue where they stopped. An answer is labelled
1 only if everything it states is correct and it answers the question; saying that the
sources do not answer a question they indeed do not answer is also correct. The confidence
is hidden while labelling, so that it cannot influence the label. An answer without
checkable claims has no raw confidence and is not used for fitting.
"""
import argparse
import json
import os
from pathlib import Path

import llm
from answering.answer import answer
from claimcal import check
from config import INDEX_DIR


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)                           # an interrupted write keeps the old file


def run(questions, path, index):
    rows = load(path)
    done = {r["question"] for r in rows}
    models = {role: llm.resolve(role)[1] for role in llm.ROLES}
    other = [r["models"] for r in rows if r["models"] != models]
    if other:
        raise ValueError(f"{path} holds answers of {other[0]}, current models are {models}; "
                         "use another file")
    todo = [q for q in dict.fromkeys(questions) if q not in done]
    for k, question in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {question}", flush=True)
        try:
            result = answer(question, index)
            checked = check.check(result["answer"], result["retrieved"], question)
        except llm.LLMError as e:                   # asked again on the next run
            print(f"  error: {e}")
            continue
        rows.append({"question": question, "answer": result["answer"], "models": models,
                     "chunks": result["retrieved"], "claims": checked["claims"],
                     "confidence": checked["raw"], "label": None})
        save(rows, path)


def label(path):
    rows = load(path)
    if not rows:
        raise ValueError(f"{path}: no answers to label")
    for row in rows:
        if row["label"] is not None:
            continue
        print(f"\n{'=' * 78}\nFrage: {row['question']}\n\n{row['answer']}\n")
        for n, chunk in enumerate(row["chunks"], 1):
            print(f"--- Quelle [{n}] {chunk['path']} {chunk.get('pages', '')}\n{chunk['text']}\n")
        key = ""
        while key not in ("r", "f", "s", "q"):
            key = input("[r]ichtig, [f]alsch, [s] überspringen, [q] beenden: ").strip().lower()
        if key == "q":
            break
        if key != "s":
            row["label"] = int(key == "r")
            save(rows, path)
    labels = [r["label"] for r in rows if r["label"] is not None]
    print(f"\n{len(labels)} von {len(rows)} gelabelt, davon {sum(labels)} richtig")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="answer and check every question")
    r.add_argument("questions")
    r.add_argument("-o", "--output", default="data/runs/runs.json")
    r.add_argument("--index", default=INDEX_DIR)
    sub.add_parser("label", help="label the collected answers").add_argument("runs")
    args = ap.parse_args()
    try:
        if args.command == "run":
            lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
            run([q.strip() for q in lines if q.strip() and not q.lstrip().startswith("#")],
                args.output, args.index)
        else:
            label(args.runs)
    except (OSError, ValueError, KeyError, llm.LLMError) as e:
        raise SystemExit(f"error: {e}")
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("\nabgebrochen, bisherige Ergebnisse sind gespeichert")


if __name__ == "__main__":
    main()
