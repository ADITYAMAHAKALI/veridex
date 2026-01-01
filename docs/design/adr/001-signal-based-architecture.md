# ADR-001: Signal-Based Architecture

## Status
Accepted

## Context
AI content detection is a rapidly evolving field. No single classifier is robust enough to handle all generative models, domains, and adversarial attacks. Monolithic classifiers are opaque and often fail silently. Users need transparency, interpretability, and the ability to customize which detection methods are used.

## Decision
We will adopt a **Signal-Based Architecture**.

*   **Decomposition:** The system is decomposed into independent "Signals".
*   **Independence:** Each Signal operates in isolation and produces its own score, confidence, and metadata.
*   **Aggregation:** A separate "Fusion" layer aggregates these signals into a final verdict.

## Consequences
### Positive
*   **Modularity:** New detection methods (e.g., a new paper comes out) can be added as just another Signal without refactoring the whole system.
*   **Robustness:** If one signal fails (e.g., perplexity is low because of good prompting), others (e.g., stylometric) might still catch it.
*   **Interpretability:** Users can see *why* a text was flagged (e.g., "High burstiness" vs "High perplexity gap").

### Negative
*   **Complexity:** Orchestrating multiple signals and fusing them is more complex than running a single `.predict()` call.
*   **Latency:** Running multiple signals (especially model-based ones) increases inference time. This is mitigated by allowing users to select/configure which signals to run.
