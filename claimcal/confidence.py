"""Turn probe verdicts into a risk per claim and a raw confidence per answer.

Follows the scoring of CLAIM-CAL (Pal 2026). Every claim is judged by several probes, each
returning one of the verdicts in VERDICTS. For a claim,

  risk = W_CONTRADICTED * share of probes that found it contradicted
       + W_UNKNOWN      * share of probes that found it neither supported nor contradicted

and the raw confidence of an answer is 1 - AGGREGATE(risks of its claims). The raw value
is not a probability; calibrate.py maps it to one.

No language model is called here, so the results are deterministic and cheap to recompute
from stored verdicts, for example with other weights.

  python -m claimcal.confidence claims.json      # claims with "verdicts" -> risks and confidence
"""
import json
import sys

VERDICTS = ("supported", "contradicted", "unknown")
W_CONTRADICTED = 0.7        # weight of a contradicting verdict (Pal 2026)
W_UNKNOWN = 0.5             # weight of an undecided verdict (Pal 2026)
AGGREGATE = "mean"          # "mean": average claim risk; "max": the riskiest claim decides


def claim_risk(verdicts, w_contradicted=W_CONTRADICTED, w_unknown=W_UNKNOWN):
    """-> risk of one claim from its verdicts, a list or a dict probe -> verdict."""
    if isinstance(verdicts, dict):
        verdicts = list(verdicts.values())
    if not verdicts:
        raise ValueError("a claim needs at least one verdict")
    unknown = [v for v in verdicts if v not in VERDICTS]
    if unknown:
        raise ValueError(f"unknown verdict {unknown[0]!r} (expected one of {VERDICTS})")
    n = len(verdicts)
    return (w_contradicted * verdicts.count("contradicted") / n
            + w_unknown * verdicts.count("unknown") / n)


def answer_confidence(risks, aggregate=AGGREGATE):
    """-> raw confidence of an answer from the risks of its claims; None without claims.

    An answer without claims (for example "the sources do not answer this") makes no
    statement that could be checked, so it has no confidence rather than full confidence.
    """
    risks = list(risks)
    if not risks:
        return None
    if aggregate == "mean":
        risk = sum(risks) / len(risks)
    elif aggregate == "max":
        risk = max(risks)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r} (expected 'mean' or 'max')")
    return 1.0 - risk


def score(claims, aggregate=AGGREGATE, **weights):
    """-> (claims, each copied with its "risk" added, raw confidence of the answer).

    Each claim is a dict with "verdicts"; the given dicts are not changed.
    """
    scored = [{**c, "risk": claim_risk(c["verdicts"], **weights)} for c in claims]
    return scored, answer_confidence((c["risk"] for c in scored), aggregate)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    try:
        with open(sys.argv[1], encoding="utf-8") as f:
            scored, conf = score(json.load(f))
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}")
    for c in scored:
        print(f"{c['risk']:.3f}  {c.get('text', '')}")
    print("raw confidence:", "none (no claims)" if conf is None else f"{conf:.3f}")
