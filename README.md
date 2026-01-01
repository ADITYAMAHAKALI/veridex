# Veridex

**A modular, probabilistic, and research-grounded AI content detection library.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Veridex is a production-ready library for detecting AI-generated content across multiple modalities: **text**, **image**, and **audio**. Unlike binary classifiers, Veridex provides probabilistic detection with confidence estimates and interpretable signals.

## ✨ Features

- 🎯 **Multi-Modal Detection**: Text, Image, and Audio deepfake detection
- 📊 **Probabilistic Outputs**: Returns probabilities and confidence scores, not just binary labels
- 🔍 **Interpretable Signals**: Exposes individual detection features for transparency
- 🧩 **Modular Architecture**: Easy to extend with new detection methods
- 🚀 **Production-Ready**: Robust error handling, graceful degradation
- 📖 **Research-Grounded**: Based on state-of-the-art papers and benchmarks

## 🚀 Quick Start

### Installation

```bash
# Install core library
pip install veridex

# Install with specific modality support
pip install veridex[text]      # Text detection
pip install veridex[audio]     # Audio detection
pip install veridex[image]     # Image detection

# Install everything
pip install veridex[text,image,audio]

# Development installation
pip install -e ".[dev]"
```

### Usage Examples

#### Text Detection

```python
from veridex.text import PerplexitySignal, BinocularsSignal

# Quick detection with perplexity
detector = PerplexitySignal()
result = detector.run("This text seems suspiciously perfect...")

print(f"AI Probability: {result.score:.2f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Perplexity: {result.metadata['mean_perplexity']:.2f}")
```

#### Audio Detection

```python
from veridex.audio import SpectralSignal

# Lightweight frequency analysis
detector = SpectralSignal()
result = detector.run("audio_sample.wav")

print(f"AI Probability: {result.score:.2f}")
print(f"Spectral Features: {result.metadata}")
```

#### Image Detection

```python
from veridex.image import FrequencyDomainSignal

# Analyze spectral anomalies
detector = FrequencyDomainSignal()
result = detector.run("suspicious_image.png")

print(f"AI Probability: {result.score:.2f}")
```

## 📦 Available Detectors

### Text Detectors

| Detector | Method | Speed | Accuracy | GPU Required |
|----------|--------|-------|----------|--------------|
| `ZlibEntropySignal` | Compression-based | Fast | Low | No |
| `PerplexitySignal` | Statistical (LLM-based) | Medium | Medium | Optional |
| `BinocularsSignal` | Contrastive Perplexity | Medium | High | Optional |

### Audio Detectors

| Detector | Method | Speed | Accuracy | GPU Required |
|----------|--------|-------|----------|--------------|
| `SpectralSignal` | Frequency Domain | Fast | Medium | No |
| `AASISTSignal` | Spectro-Temporal | Medium | High | No |
| `Wav2VecSignal` | Foundation Model | Slow | Very High | Recommended |

### Image Detectors

| Detector | Method | Speed | Accuracy | GPU Required |
|----------|--------|-------|---------|--------------|
| `FrequencyDomainSignal` | Spectral Analysis | Fast | Medium | No |
| `DIRESignal` | Diffusion Reconstruction | Slow | High | Yes |

## 🧪 Testing

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,text,audio,image]"

# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/audio/ -v

# With coverage
pytest tests/ --cov=veridex --cov-report=html
```

See [TESTING.md](TESTING.md) for detailed testing guide.

## 📚 Documentation

- **[Core Idea & Plan](core_idea_and_plan.md)** - Design philosophy and architecture
- **[Research Document](AI-Generated%20Content%20Detection%20Research.md)** - Comprehensive technical analysis
- **[Testing Guide](TESTING.md)** - How to test all metrics
- **Module READMEs**:
  - [Audio Detection](veridex/audio/README.md)
  - Examples in `examples/` directory

## 🏗️ Architecture

Veridex follows a signal-based architecture:

```
Input → Signal Extractors → Normalization → Fusion → Output
                ↓
    (Independent, Inspectable Signals)
```

Each detector:
- Inherits from `BaseSignal`
- Returns standardized `DetectionResult`
- Operates independently
- Declares its limitations explicitly

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/example/veridex.git
cd veridex

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode with all dependencies
pip install -e ".[dev,text,image,audio]"

# Run tests
pytest tests/

# Format code
black veridex/ tests/
flake8 veridex/
```

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 🔬 Research & Citations

Veridex is based on cutting-edge research in AI-generated content detection. Key methods include:

- **Binoculars**: Spotting LLMs With Binoculars (arXiv:2401.12070)
- **AASIST**: Audio Anti-Spoofing Integrated Spectro-Temporal Graph Attention
- **DIRE**: Diffusion Reconstruction Error for deepfake images
- **Wav2Vec 2.0**: Self-supervised foundation models for audio

See [AI-Generated Content Detection Research.md](AI-Generated%20Content%20Detection%20Research.md) for full references.

## ⚠️ Limitations

Veridex is a **probabilistic detection tool**, not a definitive proof system:

- ❌ Not suitable as sole evidence for legal/forensic purposes
- ❌ Cannot detect all AI-generated content with 100% accuracy
- ❌ Vulnerable to adversarial attacks and post-processing
- ⚠️ Requires regular updates as generative models improve

**Always use multiple signals and human judgment for critical decisions.**

## 🗺️ Roadmap

- [x] Text detection (Perplexity, Binoculars)
- [x] Image detection (Frequency, DIRE)
- [x] Audio detection (Spectral, AASIST, Wav2Vec)
- [ ] Video detection (rPPG, I3D)
- [ ] C2PA provenance integration
- [ ] Ensemble fusion models
- [ ] Real-time streaming detection
- [ ] Model calibration on benchmarks

## 📧 Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Built with ❤️ for transparency in the age of generative AI**
