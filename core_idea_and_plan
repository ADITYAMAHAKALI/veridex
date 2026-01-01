# AI Content Detection Library

**Requirements & Design Document**

---

## 1. Purpose & Motivation

The rapid adoption of generative models has made it increasingly difficult to distinguish between human-generated and AI-generated content. Existing detection tools often:

* Provide binary outputs without uncertainty
* Overfit to specific model families
* Fail silently under human post-editing
* Lack transparency around failure modes

This project aims to build a **modular, probabilistic, and research-grounded AI content detection library** that explicitly models uncertainty, exposes individual detection signals, and documents its limitations.

The library is designed to be:

* **Production-aware** (usable in real systems)
* **Research-friendly** (easy to experiment with new signals)
* **Honest** (no false claims of certainty)

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. Provide **probabilistic detection** of AI-generated content (not binary classification).
2. Support **multi-signal detection**, starting with text.
3. Expose **interpretable signals** contributing to the final score.
4. Explicitly model **confidence and reliability**.
5. Document **failure modes and adversarial weaknesses**.
6. Be modular and extensible across modalities (text → image → video).
7. Enable **rigorous evaluation and benchmarking**.

---

### 2.2 Non-Goals

* Achieving perfect or near-perfect detection accuracy.
* Providing legal or forensic guarantees.
* Defeating adversarial actors definitively.
* Acting as a censorship or moderation policy engine.
* Building watermarking or embedding-based generation control.

---

## 3. Target Users

* ML Engineers building moderation, trust, or verification systems
* Researchers studying AI detectability and robustness
* Platform teams needing explainable AI-origin signals
* Open-source contributors experimenting with detection methods

---

## 4. System Overview

The system is designed as a **signal-based detection framework** rather than a monolithic classifier.

High-level flow:

```
Input Content
   ↓
Signal Extractors (Independent)
   ↓
Signal Normalization
   ↓
Fusion & Calibration
   ↓
Probabilistic Output + Confidence + Failure Flags
```

Each signal:

* Operates independently
* Can be evaluated in isolation
* Declares its known limitations

---

## 5. Functional Requirements

### 5.1 Input Types

**Phase 1 (Required)**

* Plain text (UTF-8)

**Phase 2 (Planned)**

* Images (PNG, JPG)

**Phase 3 (Optional)**

* Video (frame-based analysis)

---

### 5.2 Output Schema

The system MUST return structured, interpretable output.

```json
{
  "ai_generated_probability": float,        // [0, 1]
  "confidence": float,                      // reliability estimate [0, 1]
  "signals": {
    "signal_name": float
  },
  "signal_metadata": {
    "signal_name": {
      "applicable": boolean,
      "notes": string
    }
  },
  "failure_modes": [string],
  "warnings": [string]
}
```

---

### 5.3 Detection Signals (Text)

#### Required Signal Categories

1. **Perplexity-Based Signals**

   * Single-model perplexity
   * Cross-model perplexity gap
   * Sliding-window variance

2. **Stylometric Signals**

   * Sentence length distribution
   * Burstiness and entropy
   * POS tag ratios
   * Function word frequency

3. **Compression & Redundancy Signals**

   * Compression ratio
   * n-gram repetition
   * Token saturation

Each signal MUST:

* Output a normalized score
* Declare when it is unreliable
* Be independently testable

---

### 5.4 Fusion & Scoring

The system MUST:

* Combine signals using a transparent fusion strategy
* Support weighted aggregation
* Allow calibration using held-out data
* Expose raw and calibrated scores

Initial fusion may be heuristic-based; learned fusion is optional and experimental.

---

### 5.5 Evaluation & Benchmarking

The system MUST support:

* Multiple datasets
* Human-only vs AI-only vs human-edited AI samples
* Domain-shift testing
* Stress testing (short text, paraphrasing)

Required metrics:

* AUROC
* Calibration error (ECE)
* False positive rate at fixed thresholds

---

## 6. Non-Functional Requirements

### 6.1 Explainability

* All signals must be inspectable
* No black-box-only decisions
* Clear documentation for each signal

### 6.2 Reliability & Robustness

* Graceful degradation when signals fail
* Explicit warnings for low-confidence outputs

### 6.3 Extensibility

* New signals can be added without modifying core logic
* Modalities are isolated by design

### 6.4 Performance

* Text detection should run on CPU
* Optional GPU acceleration for heavy models
* Reasonable latency for API use (<500ms for text)

---

## 7. System Architecture

### 7.1 Module Layout

```
ai_content_detection/
├── core/
│   ├── signal.py          # Base signal interface
│   ├── fusion.py          # Score aggregation logic
│   ├── calibration.py     # Confidence estimation
│
├── text/
│   ├── perplexity.py
│   ├── stylometry.py
│   ├── compression.py
│
├── image/
│   ├── frequency.py
│   ├── diffusion.py
│
├── evaluation/
│   ├── datasets.py
│   ├── metrics.py
│   ├── stress_tests.py
│
├── api/
│   ├── fastapi_app.py
│
├── docs/
│   ├── theory.md
│   ├── failure_modes.md
│   ├── adversarial.md
│
└── examples/
```

---

### 7.2 Signal Interface (Conceptual)

Each signal implements:

* `is_applicable(input)`
* `extract_features(input)`
* `score(features)`
* `known_failure_modes()`

This enforces consistency and interpretability.

---

## 8. Failure Mode Philosophy

Failure modes are **first-class outputs**, not hidden issues.

Examples:

* Short text length
* Heavy human post-editing
* Domain mismatch
* Non-native language artifacts

The system MUST surface these explicitly.

---

## 9. Security & Ethical Considerations

* Avoid claims of certainty
* Avoid misuse as sole moderation authority
* Document adversarial evasion strategies
* Prioritize false-positive minimization

---

## 10. Development Phases

### Phase 1 — Text Detection (MVP)

* Core architecture
* 3–4 signal types
* Evaluation framework
* OSS release

### Phase 2 — Image Detection

* Frequency-domain signals
* Diffusion artifact analysis

### Phase 3 — Advanced Research

* Video analysis
* Learned fusion
* Adversarial robustness studies

---

## 11. Success Criteria

The project is successful if:

* Signals are interpretable and modular
* Evaluation reveals honest limitations
* External contributors can add new signals
* The library is adopted or referenced in real systems
* The documentation demonstrates architectural maturity

---

## 12. Long-Term Vision

This library becomes:

* A trust layer for civic platforms
* A research sandbox for AI detectability
* A reference implementation for probabilistic AI detection
* A foundation for policy-aware moderation systems

---
