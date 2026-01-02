# AI Image Detection Research Summary

## Overview
This document summarizes the review of 10 key papers on AI image detection. The goal is to identify robust methods for detecting synthetic images, particularly from diffusion models, and to select promising candidates for implementation in `veridex`.

## Key Findings

### 1. Generalization is the Main Challenge
Most early detectors (trained on GANs) fail to generalize to Diffusion models (DALL-E, Stable Diffusion).
*   **Paper 9 (UnivFD)** and **Paper 6 (CLIP)** highlight that training on specific artifacts (like GAN fingerprints) leads to overfitting.
*   **Paper 2 (Sanity Check)** confirms that many detectors rely on bias rather than genuine generation artifacts.

### 2. CLIP Features are Powerful
*   **Paper 9 (UnivFD)**: Using frozen CLIP ViT features with a simple classifier (Nearest Neighbor or Linear Probe) achieves state-of-the-art generalization. It works because CLIP features are "blind" to the specific real-vs-fake task during pre-training, preventing it from latching onto fragile low-level artifacts.
*   **Paper 6 (Raising the Bar)**: Also leverages CLIP, showing it captures semantic inconsistencies or subtle artifacts better than standard ResNets.
*   **Paper 8 (C2P-CLIP)**: Improvements on CLIP by injecting category prompts.

### 3. Reconstruction and Entropy Methods
*   **Paper 5 (DIRE)**: Measures the error when reconstructing an image via a pre-trained diffusion model. High error -> Real (hard to reconstruct perfectly), Low error -> Fake (easy for the model to reproduce its own patterns). *Already implemented in Veridex.*
*   **Paper 3 (MLEP)**: Multi-granularity Local Entropy Patterns. Analyzes local texture/entropy. Good for universal detection without heavy deep learning backbones.
*   **Paper 4 (PatchCraft)**: Focuses on small texture patches to find artifacts.

## Selected Methods for Implementation

Based on performance, robustness, and architectural fit, the following methods are selected for implementation:

### 1. `CLIPSignal` (based on Paper 9 - UnivFD)
*   **Concept**: Extract image embeddings using a pre-trained CLIP model (frozen). Use a lightweight linear classifier (or simple thresholding on distance to real/fake clusters) to detect synthetic content.
*   **Why**: Best-in-class generalization. Complementary to DIRE.
*   **Implementation Plan**:
    *   Create `veridex/image/clip.py`.
    *   Use `transformers` or `open_clip` to load CLIP.
    *   Implement a linear probe or distance-based scorer.

### 2. `MLEPSignal` (based on Paper 3)
*   **Concept**: Calculate local entropy maps of the image at multiple granularities. Extract statistical features from these maps.
*   **Why**: Provides a signal based on information theory/statistics rather than deep learning semantics or reconstruction error. Lightweight and explainable.
*   **Implementation Plan**:
    *   Create `veridex/image/mlep.py`.
    *   Implement local entropy calculation using `scipy`/`numpy`.

## Existing Signals
*   `DIRESignal`: Already implemented (Paper 5).
*   `FrequencySignal`: FFT-based (related to concepts in Paper 1 and 2 regarding frequency artifacts).

## Next Steps
1.  Implement `CLIPSignal`.
2.  Implement `MLEPSignal`.
3.  Register new signals in the library.
