# Concepts

Veridex is built on a few core concepts that differentiate it from other detection libraries.

## The Signal Architecture

At the heart of Veridex is the **Signal**. A Signal is an atomic unit of detection logic. It takes a specific type of input (e.g., text string, image array) and outputs a `DetectionResult`.

### Why Signals?

- **Independence**: Each signal runs in isolation. If one fails (e.g., due to missing dependencies), it doesn't crash the pipeline.
- **Explainability**: You can inspect the output of each signal individually to understand *why* content was flagged.
- **Extensibility**: Adding a new detection method is as simple as subclassing `BaseSignal` and implementing the `run()` method.

## Probabilistic Scoring

Unlike binary classifiers that output "Fake" or "Real", Veridex signals output a **Score** and a **Confidence**.

- **Score**: A value between 0.0 (Human) and 1.0 (AI).
- **Confidence**: A value between 0.0 and 1.0 indicating how reliable the signal considers its own assessment.

## Fusion (Planned)

While individual signals are useful, they often have weaknesses. We are working on **Fusion** strategies to aggregate multiple signals into a final verdict. This will allow combining weak signals (like perplexity) with strong signals (like DetectGPT) to produce a robust final decision.

*(Fusion implementation is currently in development.)*
