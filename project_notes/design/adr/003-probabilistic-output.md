# ADR-003: Probabilistic Output & Uncertainty

## Status
Accepted

## Context
Binary classification (Real vs. Fake) is misleading in the context of AI detection. There is no ground truth "fake" in many semantic senses, and detectors often have high false positive rates on out-of-distribution data. Users need to know *how confident* the system is, not just a label.

## Decision
All Signals and the final Detector must return a **Probabilistic Result** object, not a boolean.

The schema must include:
1.  `score`: A normalized float [0.0, 1.0] indicating the "degree of AI-ness".
2.  `confidence`: A float [0.0, 1.0] indicating the reliability of the estimation (e.g., based on input length or domain match).
3.  `metadata`: A dictionary of extra info (e.g., "tokens_processed": 50).

## Consequences
### Positive
*   **Nuance:** Allows downstream systems to set their own thresholds (e.g., "Only flag if score > 0.9 AND confidence > 0.8").
*   **Safety:** Reduces the risk of blindly banning users based on a shaky prediction.

### Negative
*   **Ambiguity:** Some users just want a "Yes/No" answer. We can provide a helper method `result.is_ai(threshold=0.5)` for convenience, but the primary data model remains probabilistic.
