"""Map raw answer confidence to a calibrated probability by isotonic regression.

Isotonic regression fits the best non-decreasing step function from raw confidence to the
observed share of correct answers (pool adjacent violators). Between the fitted points the
mapping is interpolated linearly, outside them it is clipped to the end values. Only these
points are stored, so a calibration is a small JSON file:

  {"method": "isotonic", "n": 60, "x": [...], "y": [...]}

A calibration belongs to the verifier model and the probes that produced the raw values.
Without a calibration file the mapping is the identity, and the result says so.

  python -m claimcal.calibrate fit data.json -o calibration.json   # data: [{"confidence", "label"}]
  python -m claimcal.calibrate ece data.json [--calibration calibration.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import CALIBRATION_FILE

ECE_BINS = 10
FOLDS = 5


class Calibrator:
    """A monotone mapping from raw confidence to probability, given by points (x, y)."""

    def __init__(self, x=(), y=(), n=0):
        self.x, self.y, self.n = np.asarray(x, float), np.asarray(y, float), n
        if self.x.shape != self.y.shape or np.any(np.diff(self.x) < 0):
            raise ValueError("calibration points must be sorted by x, one y per x")

    @property
    def calibrated(self):
        """False for the identity mapping used when no calibration exists."""
        return len(self.x) > 0

    def apply(self, raw):
        """-> calibrated value(s) of raw confidence; None stays None."""
        if raw is None:
            return None
        if not self.calibrated:
            return raw
        out = np.interp(raw, self.x, self.y)
        return float(out) if np.ndim(out) == 0 else out

    def to_dict(self):
        return {"method": "isotonic", "n": self.n, "x": self.x.tolist(), "y": self.y.tolist()}


def fit(raw, labels):
    """-> Calibrator fitted to raw confidences and labels (1 correct, 0 not correct)."""
    raw, labels = np.asarray(raw, float), np.asarray(labels, float)
    if raw.ndim != 1 or raw.shape != labels.shape or not len(raw):
        raise ValueError("need equally many raw confidences and labels, at least one")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be 0 or 1")

    # Equal raw values cannot be ordered, so they start as one block: their mean label.
    xs, inverse = np.unique(raw, return_inverse=True)
    weight = np.bincount(inverse).astype(float)
    value = np.bincount(inverse, weights=labels) / weight

    # Pool adjacent violators: merge neighbouring blocks while a block lies above its
    # successor. Each block keeps its weighted mean, its weight and its first x index.
    blocks = []                                     # [mean, weight, first index]
    for i in range(len(xs)):
        blocks.append([value[i], weight[i], i])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, _ = blocks.pop()
            m1, w1, s1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, s1])

    # A block is flat from its first to its last x; store both ends to keep it flat.
    x, y = [], []
    for k, (m, _, start) in enumerate(blocks):
        end = blocks[k + 1][2] - 1 if k + 1 < len(blocks) else len(xs) - 1
        for i in sorted({start, end}):
            x.append(xs[i])
            y.append(m)
    return Calibrator(x, y, n=len(raw))


def save(calibrator, path=CALIBRATION_FILE):
    Path(path).write_text(json.dumps(calibrator.to_dict(), indent=1), encoding="utf-8")


def load(path=CALIBRATION_FILE):
    """-> the stored Calibrator, or the identity if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return Calibrator()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("method") != "isotonic":
        raise ValueError(f"{path}: unknown calibration method {data.get('method')!r}")
    return Calibrator(data["x"], data["y"], data.get("n", 0))


def ece(confidences, labels, bins=ECE_BINS):
    """-> Expected Calibration Error over equal-width confidence bins.

    The sum over bins of (share of answers in the bin) * |mean confidence - accuracy|.
    """
    conf, labels = np.asarray(confidences, float), np.asarray(labels, float)
    if conf.shape != labels.shape or not len(conf):
        raise ValueError("need equally many confidences and labels, at least one")
    # Bin b holds (b/bins, (b+1)/bins]; confidence 0 goes to the first bin.
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - labels[mask].mean())
    return float(total)


def cross_validate(raw, labels, folds=FOLDS):
    """-> calibrated confidence of each answer, from a fit that did not see that answer."""
    raw, labels = np.asarray(raw, float), np.asarray(labels, float)
    fold = np.random.default_rng(0).permutation(len(raw)) % folds
    out = np.empty(len(raw))
    for k in range(folds):
        test = fold == k
        if test.any():
            out[test] = fit(raw[~test], labels[~test]).apply(raw[test])
    return out


def _read(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("confidence") is not None and r.get("label") is not None]
    return [r["confidence"] for r in rows], [r["label"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="fit a calibration to labelled raw confidences")
    f.add_argument("data")
    f.add_argument("-o", "--output", default=CALIBRATION_FILE)
    e = sub.add_parser("ece", help="expected calibration error of labelled confidences")
    e.add_argument("data")
    e.add_argument("--calibration", help="apply this calibration first")
    args = ap.parse_args()
    try:
        raw, labels = _read(args.data)
        if args.command == "fit":
            cal = fit(raw, labels)
            save(cal, args.output)
            print(f"{args.output}: fitted on {cal.n} answers ({int(sum(labels))} correct), "
                  f"{len(cal.x)} points")
            # The ECE on the fitted answers themselves is close to 0 by construction.
            if cal.n >= 2 * FOLDS:
                print(f"ECE raw {ece(raw, labels):.3f}, calibrated "
                      f"{ece(cross_validate(raw, labels), labels):.3f} "
                      f"({FOLDS}-fold cross-validation)")
            if len(set(labels)) < 2:
                print("warning: all labels are equal, the mapping is constant")
        else:
            conf = load(args.calibration).apply(raw) if args.calibration else raw
            print(f"ECE {ece(conf, labels):.3f} over {len(raw)} answers")
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
