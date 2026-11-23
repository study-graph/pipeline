"""Estimate how far an answer is supported by its sources (CLAIM-CAL, Pal 2026).

Chains claims.py -> verify.py -> confidence.py -> calibrate.py. Only the answer and its
sources are needed, not the index, so stored or benchmark answers can be checked too.
"""
import calibrate
import claims
import confidence
import verify
from config import CALIBRATION_FILE


def check(answer, chunks, question=None, method="llm", scope="cited", batch="probe",
          calibration=CALIBRATION_FILE):
    """-> dict with the judged claims, "raw" confidence, the "confidence" after calibration
    and whether it is "calibrated". Both values are None if the answer states no claims.

    chunks are the sources in the numbering of the answer: [n] is chunks[n - 1].
    """
    calibrator = calibrate.load(calibration)        # fails early, before any model call
    found = claims.decompose(answer, question, method)
    judged = verify.verify(found, chunks, scope=scope, batch=batch)
    scored, raw = confidence.score(judged)
    return {"claims": scored, "raw": raw, "confidence": calibrator.apply(raw),
            "calibrated": calibrator.calibrated}
