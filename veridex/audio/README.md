# Audio Detection Module - Quick Start

## Installation

```bash
# Install veridex with audio support
pip install veridex[audio]
```

This installs:
- torch, torchaudio (deep learning framework)
- transformers (for Wav2Vec models)
- librosa (audio processing)
- soundfile (audio I/O)

## Quick Usage

### Spectral Detector (Lightweight)

```python
from veridex.audio import SpectralSignal

detector = SpectralSignal()
result = detector.run("audio.wav")

print(f"AI Probability: {result.score:.2f}")
print(f"Confidence: {result.confidence:.2f}")
```

### Wav2Vec Detector (High Accuracy)

```python
from veridex.audio import Wav2VecSignal

detector = Wav2VecSignal(
    model_id="nii-yamagishilab/wav2vec-large-anti-deepfake"
)
result = detector.run("audio.wav")

print(f"AI Probability: {result.score:.2f}")
```

### AASIST Detector (Balanced)

```python
from veridex.audio import AASISTSignal

detector = AASISTSignal()
result = detector.run("audio.wav")

print(f"AI Probability: {result.score:.2f}")
print(f"Features: {result.metadata}")
```

## Module Structure

```
veridex/audio/
├── __init__.py           # Exports all detectors
├── utils.py              # Audio loading & preprocessing
├── spectral.py           # Frequency domain analysis
├── wav2vec_detector.py   # Foundation model detector
└── aasist_detector.py    # Spectro-temporal analysis
```

## Running Tests

```bash
pytest tests/audio/ -v
```

## Documentation

See:
- Full walkthrough: `walkthrough.md`
- Examples: `examples/audio_detection_example.py`
- Implementation plan: `implementation_plan.md`
