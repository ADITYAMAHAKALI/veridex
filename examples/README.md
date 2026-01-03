# Veridex Examples

This directory contains practical examples for using veridex detection capabilities across all modalities.

## Available Examples

### 1. Text Detection
**File:** [text_detection_example.py](text_detection_example.py)

Examples include:
- **Zlib Entropy** - Lightweight compression-based detection (no dependencies)
- **Stylometric Analysis** - Linguistic feature analysis (vocabulary richness)
- **Perplexity & Burstiness** - Statistical analysis with GPT-2
- **Binoculars** - Advanced contrastive perplexity (requires heavy models)
- **Multi-detector Ensemble** - Combining multiple signals
- **Batch Analysis** - Processing multiple text samples

**Run:**
```bash
python examples/text_detection_example.py
```

### 2. Image Detection
**File:** [image_detection_example.py](image_detection_example.py)

Examples include:
- **Frequency Domain** - Fast spectral analysis
- **ELA** - Error Level Analysis for manipulation detection
- **DIRE** - Diffusion reconstruction error (requires GPU)
- **Batch Image Analysis** - Processing multiple images
- **Image Preprocessing** - Best practices for clean detection
- **Real vs AI Comparison** - Understanding detection patterns

**Run:**
```bash
# Requires image dependencies
pip install veridex[image]
python examples/image_detection_example.py
```

### 3. Audio Detection
**File:** [audio_detection_example.py](audio_detection_example.py)

Examples include:
- **Spectral Detection** - Lightweight frequency analysis
- **Silence Analysis** - Pause pattern detection
- **AASIST** - Spectro-temporal features
- **Wav2Vec 2.0** - Foundation model detection
- **Ensemble Detection** - Combining multiple audio detectors
- **Batch Processing** - Analyzing multiple audio files

**Run:**
```bash
# Requires audio dependencies
pip install veridex[audio]
python examples/audio_detection_example.py
```

### 4. Advanced Multimodal Detection
**File:** [advanced_multimodal_example.py](advanced_multimodal_example.py)

Demonstrates advanced and experimental signals:
- **Text**: DetectGPT, T-Detect, HumanOOD (Zero-shot methods)
- **Image**: CLIP Zero-Shot, MLEP (Entropy Patterns)
- **Audio**: Breathing Pattern Analysis

**Run:**
```bash
# Requires all dependencies
pip install veridex[text,image,audio]
python examples/advanced_multimodal_example.py
```

## Quick Start

### Install Dependencies

```bash
# Install all examples dependencies
pip install veridex[text,image,audio]

# Or install specific modalities
pip install veridex[text]    # Text only
pip install veridex[audio]   # Audio only
pip install veridex[image]   # Image only
```

### Run Examples

```bash
# All examples work independently
python examples/text_detection_example.py
python examples/image_detection_example.py
python examples/audio_detection_example.py
```

## Example Output

### Text Detection
```
AI Probability: 0.82
Confidence: 0.65
Metrics:
  Mean Perplexity: 12.34
  Burstiness: 2.15
  
💡 Interpretation:
   Low perplexity suggests AI-generated text
   Low burstiness (uniform) suggests AI-generated
```

### Audio Detection
```
AI Probability: 0.73
Confidence: 0.80
Spectral Features:
  high_freq_energy: 3.42
  spectral_rolloff: 3850.00
  
💡 Analysis:
   Lower high-frequency energy suggests synthetic audio
```

### Image Detection
```
AI Probability: 0.88
Confidence: 0.75

💡 Analysis:
   High probability of AI generation detected
   Check for anomalous frequency patterns
```

## Customization

Each example can be customized:

```python
# Text: Use different models
from veridex.text import PerplexitySignal
detector = PerplexitySignal(model_id="gpt2-large")

# Audio: Adjust parameters
from veridex.audio import SpectralSignal
detector = SpectralSignal(
    target_sr=22050,
    n_fft=4096
)

# Image: Use different diffusion models
from veridex.image import DIRESignal
detector = DIRESignal(
    model_id="stabilityai/stable-diffusion-2-1",
    timestep=50
)
```

## Troubleshooting

### Missing Dependencies

If you see `ImportError` or dependency errors:

```bash
# Check which extras you have installed
pip list | grep veridex

# Install missing extras
pip install veridex[text,audio,image]
```

### Model Download Issues

Some examples download large models:
- **Perplexity (GPT-2)**: ~500MB
- **Binoculars**: ~7GB
- **Wav2Vec**: ~1.2GB
- **Stable Diffusion**: ~4GB

Set cache directory if needed:
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### GPU Requirements

Some detectors benefit from GPU:
- `DIRESignal` (image) - Highly recommended
- `Wav2VecSignal` (audio) - Recommended
- `PerplexitySignal` (text) - Optional

CPU will work but may be slow.

## Next Steps

1. **Try examples** with your own files
2. **Combine detectors** for ensemble detection
3. **Integrate into your application**
4. **Read documentation** for advanced usage

See [README.md](../README.md) for more information.

## Jupyter Notebooks

For interactive tutorials and deep dives, check out the `examples/notebooks/` directory:

1. **[Text Detection Deep Dive](notebooks/text_detection_deep_dive.ipynb)**
   - Hands-on with Perplexity, Burstiness, and Binoculars.
2. **[Image Forensics Analysis](notebooks/image_forensics_analysis.ipynb)**
   - Visualizing frequency artifacts and DIRE heatmaps.
3. **[Audio Synthetic Detection](notebooks/audio_synthetic_detection.ipynb)**
   - Analyzing spectral features and using Wav2Vec.
