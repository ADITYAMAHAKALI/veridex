# Veridex: Design Plan & Research

## 1. Overview
Veridex is a modular, probabilistic, and research-grounded AI content detection library. Unlike binary classifiers, Veridex focuses on **signal-based detection**, providing interpretable scores, confidence intervals, and failure mode analysis.

## 2. Research & Literature Review

### 2.1 Text Modality (Zero-Shot & Statistical)
To ensure the library is lightweight and effective without training massive supervised classifiers, we focus on **Zero-Shot** methods that leverage existing LLMs or statistical properties.

#### **Selected Papers for Implementation:**

1.  **Binoculars: Zero-Shot Detection of LLM-Generated Text** (Hans et al., 2024)
    *   **Core Idea:** Compares two different perplexity scores (PPL) from the same model (or two models) to form a "scoring" metric.
    *   **Why:** State-of-the-art performance for zero-shot detection. Highly effective and relatively efficient.
    *   **Implementation:** Requires a scoring model (e.g., Llama-2-7b or smaller specialized models).
    *   **Type:** `Perplexity-Based`

2.  **Fast-DetectGPT** (Bao et al., 2023)
    *   **Core Idea:** Uses the local curvature of the log-probability function. It replaces the expensive perturbation steps of original DetectGPT with a faster approximation.
    *   **Why:** Faster than the original DetectGPT while maintaining high accuracy.
    *   **Type:** `Probability-Curvature`

3.  **GLTR: Giant Language Model Test Room** (Gehrmann et al., 2019)
    *   **Core Idea:** Visualizes the ranking of tokens. AI text tends to sample from the "head" (top-k) of the distribution more consistently than humans.
    *   **Why:** Excellent for **interpretability** and "Stylometric" analysis (burstiness, top-k density).
    *   **Type:** `Stylometric / Statistical`

### 2.2 Image Modality (Artifacts & Reconstruction)

#### **Selected Papers for Implementation:**

1.  **DIRE: Diffusion Reconstruction Error** (Wang et al., 2023)
    *   **Core Idea:** Diffusion models can reconstruct images they generated better than real images. By passing an image through a diffusion inversion/reconstruction process, the error (residual) reveals its origin.
    *   **Why:** Strong generalized detection for Diffusion models (Stable Diffusion, Midjourney).
    *   **Type:** `Reconstruction-Based`

2.  **CNN-generated images are surprisingly easy to spot... for now** (Wang et al., 2020)
    *   **Core Idea:** Generative models leave distinct fingerprints in the frequency domain (Fourier Transform). A classifier trained on these spectral artifacts generalizes well.
    *   **Why:** Lightweight (ResNet-50 backbone) and complements DIRE. Good for GANs and older models, but also captures some diffusion artifacts.
    *   **Type:** `Frequency-Analysis`

## 3. Architecture Design

### 3.1 Core Concepts
*   **Signal:** An atomic detection unit (e.g., "Perplexity Signal", "Frequency Artifact Signal").
*   **Probe:** A wrapper that runs a Signal on an input.
*   **Result:** A probabilistic output containing `score`, `confidence`, `metadata`, and `failure_modes`.

### 3.2 Module Structure
```
veridex/
├── core/
│   ├── signal.py       # Base Abstract Class
│   ├── fusion.py       # Aggregation Logic (Weighted Avg, Logistic Regression)
│   └── calibration.py  # Calibration utilities (Platt scaling, Isotonic)
├── text/
│   ├── binoculars.py   # Binoculars implementation
│   ├── gltr.py         # GLTR-style statistical features
│   └── entropy.py      # Simple entropy/burstiness signals
├── image/
│   ├── dire.py         # Diffusion Reconstruction Error wrapper
│   └── frequency.py    # FFT-based artifact detection
└── utils/
    └── model_loader.py # Lazy loading of heavy dependencies
```

### 3.3 Dependency Management (Optional Modules)
To keep the core "lightweight", heavy libraries are optional:
*   `pip install veridex` -> Core only (numpy, scipy).
*   `pip install veridex[text]` -> Adds `transformers`, `torch`.
*   `pip install veridex[image]` -> Adds `diffusers`, `torch`, `torchvision`, `pillow`.

## 4. Implementation Strategy

### Phase 1: Foundation (Current)
*   Define `Signal` interface.
*   Define `DetectionResult` schema.
*   Setup optional dependency checking.

### Phase 2: Text Signals
*   Implement `EntropySignal` (Lightweight).
*   Implement `BinocularsSignal` (Requires `transformers`).

### Phase 3: Image Signals
*   Implement `FrequencySignal` (Lightweight-ish, requires `cv2`/`numpy`).
*   Implement `DIRESignal` (Requires `diffusers`).
