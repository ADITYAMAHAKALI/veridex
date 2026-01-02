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
* Audio (waveform/spectrogram analysis)

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

#### 1. Statistical & Zero-Shot Methods
*   **Perplexity & Burstiness**: The baseline metric. Measures "surprise" (entropy) and its variance over sentences. Low perplexity + low burstiness = high AI probability.
*   **Binoculars**: A contrastive method using two models (Performer vs. Observer) to calculate a score ratio. State-of-the-art zero-shot detection.
*   **Fast-DetectGPT**: Analyzes the conditional probability curvature of the text. Efficient zero-shot detection.
*   **Zlib Entropy**: A simple compression-based baseline.

#### 2. Watermarking & Active Detection
*   **SynthID**: Detects watermarks embedded by Google DeepMind's models (Gemini, etc.) via logits processing.

#### 3. Stylometric Signals
*   **Sentence length distribution**
*   **POS tag ratios**

---

### 5.4 Detection Signals (Image)

#### 1. Frequency Domain
*   **Spectral Analysis**: Detecting anomalous high-frequency decay (roll-off) typical of diffusion upsampling.
*   **FreqCross**: Fusing spatial and frequency features (Future).

#### 2. Reconstruction & Artifacts
*   **DIRE (Diffusion Reconstruction Error)**: Measures the error when re-passing the image through a diffusion model. AI images reconstruct better (lower error).

#### 3. Watermarking
*   **Invisible-Watermark**: Decoding DWT/DCT watermarks (e.g., Stable Diffusion tags).
*   **Stable Signature**: Latent watermarking detection.

---

### 5.5 Provenance (All Modalities)

*   **C2PA Manifests**: Cryptographically verifying the history of the asset using `c2pa-python`. This is the "Gold Standard" for provenance.

---

### 5.6 Fusion & Scoring

The system MUST:

* Combine signals using a transparent fusion strategy
* Support weighted aggregation
* Allow calibration using held-out data
* Expose raw and calibrated scores

Initial fusion may be heuristic-based; learned fusion is optional and experimental.

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

* Text detection should run on CPU where possible.
* Heavy models (Binoculars, DIRE) should be optional dependencies.
* Reasonable latency for API use (<500ms for lightweight, <5s for heavy).

---

## 7. System Architecture

### 7.1 Module Layout

```
veridex/
├── core/
│   ├── signal.py          # Base signal interface
│   ├── fusion.py          # Score aggregation logic
│   ├── calibration.py     # Confidence estimation
│   ├── provenance.py      # C2PA and metadata checks
│
├── text/
│   ├── perplexity.py      # Perplexity + Burstiness
│   ├── binoculars.py      # Binoculars (Contrastive)
│   ├── fast_detectgpt.py  # Fast-DetectGPT
│   ├── synthid.py         # SynthID wrapper
│   ├── entropy.py         # Zlib entropy
│
├── image/
│   ├── frequency.py       # Spectral analysis
│   ├── dire.py            # Diffusion Reconstruction
│   ├── watermarks.py      # Invisible-Watermark / Stable Signature
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

### Phase 1 — Text Detection Core (MVP)

* Core architecture (Done)
* Basic Signals: Zlib Entropy, Perplexity (Done)
* **New**: Burstiness (Upgrade Perplexity)
* **New**: Binoculars (High fidelity zero-shot)

### Phase 2 — Image & Provenance

* Image Frequency Analysis (Done)
* DIRE (Done)
* **New**: C2PA Integration (Provenance)
* **New**: Image Watermarking decoding

### Phase 3 — Advanced Text & Video

* Fast-DetectGPT
* SynthID Integration
* Video rPPG (Physiological signals)

### Phase 4 — Research & Hardening

* Adversarial robustness studies
* Learned fusion models
* Audio detection (AASIST)

---

## 11. Success Criteria

The project is successful if:

* Signals are interpretable and modular
* Evaluation reveals honest limitations
* External contributors can add new signals
* The library is adopted or referenced in real systems
* The documentation demonstrates architectural maturity

---
