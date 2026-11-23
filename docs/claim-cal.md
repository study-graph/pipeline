# Confidence scoring (CLAIM-CAL)

`ask --verify` follows CLAIM-CAL (Pal 2026): `claimcal/check.py` splits the answer into atomic
claims, checks each claim against its cited sources with several probes, and turns the
verdicts into a confidence.

| Module | What it does |
| --- | --- |
| `claimcal/claims.py` | Splits an answer into claims, each with the source numbers the answer cites for it. Method `llm` (default) lets the verify model form atomic, self-contained claims; method `sentences` takes every sentence or list item as one claim, without a model call. Source numbers the answer does not cite are dropped. |
| `claimcal/verify.py` | Judges every claim against the sources it cites with three probes (support-seeking, balanced, contradiction-seeking), each answering `supported`, `contradicted` or `unknown`. By default one model call per probe covers all claims. A claim without a cited source is not sent to the model and counts as `unknown`. |
| `claimcal/confidence.py` | Risk per claim from the share of contradicting and undecided verdicts (weights 0.7 and 0.5); raw answer confidence is 1 minus the mean claim risk. |
| `claimcal/calibrate.py` | Fits an isotonic regression from raw confidence to the observed share of correct answers, stores it as `calibration.json`, and computes the Expected Calibration Error. Without the file, confidence stays uncalibrated. |

```sh
uv run python -m claimcal.claims answer.txt --question "..."            # JSON list of claims
uv run python -m claimcal.calibrate fit labelled.json   # [{"confidence", "label"}]
uv run python -m claimcal.calibrate ece heldout.json --calibration calibration.json
```

Fitted on some answers, the ECE on these same answers is close to zero by construction;
only the ECE on held-out answers measures calibration.

## Calibration

`claimcal/collect.py` collects the labelled answers that the fit needs:

```sh
uv run python -m claimcal.collect run data/questions.txt        # answers and raw confidence
uv run python -m claimcal.collect label data/runs/runs.json     # judge each answer
uv run python -m claimcal.calibrate fit data/runs/runs.json     # writes calibration.json
```

The first two steps continue where they stopped. `fit` reports the ECE by 5-fold
cross-validation. A calibration belongs to the models, prompts and probes it was fitted
with; after changing one of them, fit again on newly checked answers.
