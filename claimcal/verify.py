"""Check claims against their sources with several differently framed probes.

Follows CLAIM-CAL (Pal 2026) with two changes. A claim is checked against the lecture
material it cites, not against the model's own knowledge. And a probe does not see the
full answer: shown the answer, the model takes it as evidence and confirms invented claims
(qwen3.6, batch "probe": 5 of 5 invented claims supported, 0 of 5 without). Each probe judges each claim
as one of

  supported     the sources state the claim
  contradicted  the sources state something incompatible with the claim
  unknown       the sources do not decide the claim

The probes differ only in framing (PROBES): one looks for support, one judges neutrally,
one looks for contradictions. confidence.py turns the verdicts into a risk.

Settings, each a parameter of verify():
  scope   "cited": a claim is judged by the sources it cites (default);
          "retrieved": by all retrieved sources, so a wrongly cited claim can still pass.
  batch   "probe": one model call per probe with all claims (3 calls per answer);
          "claim": one call per probe and claim (3 x claims calls, independent judgements).
Up to WORKERS calls run at a time.

  python -m claimcal.verify claims.json chunks.json      # claims from claims.py, chunks as retrieved
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import llm
from claimcal.confidence import VERDICTS

# Parallel calls help with a remote API. A local server shares one GPU between them, so
# there they are slower than sequential calls (measured with Ollama: 34.6 s against 23.5 s).
WORKERS = 1
SCOPES = ("cited", "retrieved")
BATCHES = ("probe", "claim")

COMMON = (
    "Beurteile Aussagen ausschließlich anhand der angegebenen Quellen, nicht anhand deines "
    "eigenen Wissens. Urteile pro Aussage mit genau einem Wort: \"supported\", wenn die "
    "Quellen die Aussage belegen; \"contradicted\", wenn die Quellen etwas angeben, das mit "
    "der Aussage unvereinbar ist; \"unknown\", wenn die Quellen die Aussage weder belegen "
    "noch ihr widersprechen. Beurteile jede Aussage nur anhand der Quellen, die bei ihr "
    "angegeben sind."
)
PROBES = {
    "support": "Suche gezielt nach Textstellen, die die Aussage belegen.",
    "balanced": "Wäge neutral ab, ob die Quellen die Aussage belegen, ihr widersprechen "
                "oder keine Entscheidung erlauben.",
    "contradiction": "Suche gezielt nach Textstellen, die der Aussage widersprechen oder sie "
                     "einschränken, und prüfe Details wie Zahlen, Reihenfolgen und Namen.",
}
FORMAT = 'Format: {"verdicts": [{"claim": 1, "verdict": "supported"}, ...]}'


def evidence(claim, n_chunks, scope):
    """-> the source numbers a claim is judged by."""
    if scope == "cited":
        return [n for n in claim["sources"] if 1 <= n <= n_chunks]
    if scope == "retrieved":
        return list(range(1, n_chunks + 1))
    raise ValueError(f"unknown scope {scope!r} (expected one of {SCOPES})")


def build_prompt(items, chunks):
    """User prompt for one call. items: [(claim number, claim, source numbers)]."""
    used = sorted({n for _, _, numbers in items for n in numbers})
    sources = "\n\n".join(f"[{n}] {chunks[n - 1]['text']}" for n in used)
    claims = "\n".join(f"{i}. {c['text']} (Quellen: {', '.join(f'[{n}]' for n in numbers)})"
                       for i, c, numbers in items)
    return f"Quellen:\n{sources}\n\nAussagen:\n{claims}"


def ask_probe(probe, items, chunks):
    """One model call. -> {claim number: verdict} for the claims in items it judged."""
    system = f"{COMMON} {PROBES[probe]}\n\n{FORMAT}"
    data = llm.generate_json(system, build_prompt(items, chunks), role="verify")
    rows = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise llm.FormatError(f'{probe}: response is not of the form {{"verdicts": [...]}}')
    wanted, out = {i for i, _, _ in items}, {}
    for row in rows:
        try:
            i, verdict = int(row["claim"]), str(row["verdict"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        if i in wanted and verdict in VERDICTS:
            out[i] = verdict
    return out


def judge(probe, items, chunks):
    """ask_probe, asking again for what is missing. -> {claim number: verdict}

    Claims left out of a response are asked again on their own; if a response is unusable
    as a whole, its claims are asked in two halves. A claim that fails alone stays missing.
    """
    try:
        out = ask_probe(probe, items, chunks)
    except llm.FormatError:
        out = {}
    missing = [item for item in items if item[0] not in out]
    if not missing or len(items) == 1:
        return out
    half = (len(missing) + 1) // 2
    for part in [missing] if len(missing) < len(items) else [missing[:half], missing[half:]]:
        out.update(judge(probe, part, chunks))
    return out


def verify(claims, chunks, probes=tuple(PROBES), scope="cited", batch="probe",
           workers=WORKERS):
    """-> claims, each copied with "verdicts" {probe: verdict} and "evidence" added.

    chunks are the sources in the numbering of the answer: [n] is chunks[n - 1].
    A claim without evidence is not sent to the model; every probe counts it as
    "unknown". A verdict the model still does not give after being asked again also counts
    as "unknown", and the probe is listed under "failed" of that claim.
    """
    if batch not in BATCHES:
        raise ValueError(f"unknown batch {batch!r} (expected one of {BATCHES})")
    unknown_probes = [p for p in probes if p not in PROBES]
    if unknown_probes:
        raise ValueError(f"unknown probe {unknown_probes[0]!r} (expected from {list(PROBES)})")
    items = [(i, c, evidence(c, len(chunks), scope)) for i, c in enumerate(claims, 1)]
    checkable = [item for item in items if item[2]]
    groups = [checkable] if batch == "probe" else [[item] for item in checkable]
    jobs = [(p, g) for p in probes for g in groups if g]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda job: (job[0], judge(job[0], job[1], chunks)), jobs))
    found = {p: {} for p in probes}
    for probe, verdicts in results:
        found[probe].update(verdicts)

    out = []
    for i, claim, numbers in items:
        verdicts = {p: found[p].get(i, "unknown") for p in probes}
        failed = [p for p in probes if numbers and i not in found[p]]
        out.append({**claim, "evidence": numbers, "verdicts": verdicts,
                    **({"failed": failed} if failed else {})})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("claims", help="JSON list of claims, as written by claims.py")
    ap.add_argument("chunks", help="JSON list of the retrieved chunks, in answer order")
    ap.add_argument("--scope", choices=SCOPES, default="cited")
    ap.add_argument("--batch", choices=BATCHES, default="probe")
    args = ap.parse_args()
    try:
        with open(args.claims, encoding="utf-8") as f:
            claims = json.load(f)
        with open(args.chunks, encoding="utf-8") as f:
            chunks = json.load(f)
        result = verify(claims, chunks, scope=args.scope, batch=args.batch)
    except (OSError, ValueError, KeyError, llm.LLMError) as e:
        raise SystemExit(f"error: {e}")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
