# Seeing Through the Noise: Detecting AI Images

As AI image generators like Midjourney, DALL-E 3, and Stable Diffusion become more photorealistic, distinguishing them from real photos is becoming increasingly difficult. However, these models leave behind invisible artifacts—mathematical traces of their generation process.

This post explains how **Veridex** helps you uncover these hidden signals.

## 1. Frequency Analysis

When you look at an image, you see pixels. But computers can also look at an image in the **frequency domain**.

Real cameras capture light in a way that produces specific frequency distributions. AI models, particularly those based on Convolutional Neural Networks (CNNs) or Diffusion processes, often struggle to reproduce these natural statistics perfectly. They may introduce "checkerboard artifacts" or unusual spikes in the frequency spectrum.

**Veridex**'s `FrequencySignal` uses Fourier Transforms (FFT) to analyze the image's spectral components.

```python
from veridex.image import FrequencySignal

detector = FrequencySignal()
# Returns a score based on deviation from natural image statistics
result = detector.detect("path/to/suspicious_image.jpg")
```

## 2. Diffusion Reconstruction Error (DIRE)

Most modern AI image generators are **Diffusion Models**. They work by gradually removing noise from a random starting point to form an image.

A clever way to detect them is to check if an image is "too easy" to reconstruct. The **DIRE** method (Diffusion Reconstruction Error) works like this:
1. Take an input image.
2. Add a little bit of noise to it.
3. Ask a diffusion model to reconstruct the original image.
4. Compare the reconstruction to the original.

**Logic:** If the image was created by a diffusion model, the model "knows" it well and can reconstruct it very accurately (low error). If it's a real photo, the model will struggle more (high error).

**Veridex** implements this in `DIRESignal`:

```python
from veridex.image import DIRESignal

# Uses Stable Diffusion v1.5 by default
detector = DIRESignal(device="cuda") # GPU recommended!
result = detector.detect("path/to/image.png")

print(f"AI Probability: {result.score}")
```

## 3. Local Entropy & Statistical Artifacts

Sometimes, the clues are in the pixel statistics. AI images can have inconsistent noise patterns or "too smooth" textures in certain areas.

**Veridex** provides `MLEPSignal` (Maximum Local Entropy Probability) to analyze local texture complexity.

## Getting Started

Image detection often requires heavy libraries (like `torch` and `diffusers`). Veridex keeps these optional.

To get started:
```bash
pip install veridex[image]
```

Then check out the [Image Forensics Notebook](../../examples/notebooks/image_forensics_analysis.ipynb) to see these detectors in action.
