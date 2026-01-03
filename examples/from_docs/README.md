# Examples from Documentation

This directory contains all Python code examples extracted from the Veridex documentation. These examples are organized for easy testing and reference.

## Directory Structure

```
from_docs/
├── tutorials/          # Examples from tutorial documentation
│   ├── quick_start_examples.py
│   ├── text_detection_examples.py
│   ├── image_detection_examples.py
│   ├── audio_detection_examples.py
│   └── ensemble_examples.py
└── use_cases/          # Real-world use case implementations
    ├── content_moderator.py
    ├── essay_checker.py
    ├── fact_checker.py
    ├── forensic_analyzer.py
    ├── compliance_scanner.py
    └── dataset_curator.py
```

## Tutorial Examples

### Quick Start Examples
**File:** `tutorials/quick_start_examples.py`  
**Source:** `docs/tutorials/quick_start.md`  
**Count:** 10 examples

- Basic text, image, and audio detection
- Understanding DetectionResult structure
- Exploring metadata across modalities
- Comparing multiple detectors

### Text Detection Examples
**File:** `tutorials/text_detection_examples.py`  
**Source:** `docs/tutorials/text_detection_guide.md`  
**Count:** 10 examples

- All 4 text signal types (Perplexity, Binoculars, ZlibEntropy, Stylometric)
- Practical examples: student essays, batch processing, ensemble approach
- Helper functions: preprocessing, short text handling, comprehensive checking

### Image Detection Examples
**File:** `tutorials/image_detection_examples.py`  
**Source:** `docs/tutorials/image_detection_guide.md`  
**Count:** 3 examples

- FrequencyDomainSignal (quick screening)
- DIRESignal (high accuracy, requires GPU)
- ELASignal (manipulation detection)

### Audio Detection Examples
**File:** `tutorials/audio_detection_examples.py`  
**Source:** `docs/tutorials/audio_detection_guide.md`  
**Count:** 4 examples

- SpectralSignal (fastest)
- AASISTSignal (anti-spoofing)
- Wav2VecSignal (highest accuracy)
- SilenceSignal (pattern analysis)

### Ensemble Examples
**File:** `tutorials/ensemble_examples.py`  
**Source:** `docs/tutorials/ensemble_detection.md`  
**Count:** 5 examples

- Simple averaging
- Weighted ensemble
- Confidence-based voting
- Multi-modal detection
- Production pipeline with ProductionDetector class

## Use Case Examples

### 1. Content Moderator
**File:** `use_cases/content_moderator.py`  
**Use Case:** Social media content moderation  
**Class:** `ContentModerator`

Detects AI-generated content in social media posts (text + images) and flags for review.

### 2. Essay Checker
**File:** `use_cases/essay_checker.py`  
**Use Case:** Academic integrity  
**Class:** `EssayChecker`

Analyzes student essays using ensemble of high-accuracy detectors with conservative thresholds.

### 3. Fact Checker
**File:** `use_cases/fact_checker.py`  
**Use Case:** Journalism and fact-checking  
**Class:** `FactChecker`

Multi-modal verification for news articles (text, images, audio) with risk assessment.

### 4. Forensic Analyzer
**File:** `use_cases/forensic_analyzer.py`  
**Use Case:** Legal discovery  
**Class:** `ForensicAnalyzer`

Batch document analysis for law firms with two-stage detection and reporting.

### 5. Compliance Scanner
**File:** `use_cases/compliance_scanner.py`  
**Use Case:** Enterprise compliance  
**Class:** `ComplianceScanner`

Ensures marketing content has proper AI disclosure when AI-generated content is detected.

### 6. Dataset Curator
**File:** `use_cases/dataset_curator.py`  
**Use Case:** Research data curation  
**Class:** `DatasetCurator`

Filters and categorizes dataset samples into human/AI/uncertain buckets.

## Running the Examples

### Prerequisites

```bash
# Install dependencies based on what you want to run
pip install veridex[text,image,audio]  # All modalities
# OR
pip install veridex[text]   # Text only
pip install veridex[image]  # Image only
pip install veridex[audio]  # Audio only
```

### Run Individual Examples

Each file can be run directly:

```bash
# Tutorial examples
python examples/from_docs/tutorials/quick_start_examples.py
python examples/from_docs/tutorials/text_detection_examples.py
python examples/from_docs/tutorials/ensemble_examples.py

# Use case examples
python examples/from_docs/use_cases/content_moderator.py
python examples/from_docs/use_cases/essay_checker.py
```

### Import as Modules

You can also import specific examples:

```python
from examples.from_docs.tutorials.text_detection_examples import example_1_perplexity_signal
from examples.from_docs.use_cases.essay_checker import EssayChecker

# Use the imported examples
example_1_perplexity_signal()
checker = EssayChecker()
```

## Using for Testing

These examples are ideal for creating tests:

```python
# In your test file
def test_quick_start_text_detection():
    """Test basic text detection from docs works"""
    from examples.from_docs.tutorials.quick_start_examples import example_1_basic_text_detection
    
    # Should not raise any exceptions
    example_1_basic_text_detection()

def test_essay_checker_class():
    """Test EssayChecker use case implementation"""
    from examples.from_docs.use_cases.essay_checker import EssayChecker
    
    checker = EssayChecker()
    result = checker.analyze_essay("Sample essay text...")
    
    assert 'status' in result
    assert 'confidence' in result
```

## Notes

### Sample Data Requirements

Some examples require external files:

- **Image examples**: Need actual image files (PNG, JPG, etc.)
- **Audio examples**: Need audio files (WAV format recommended)
- **Use case examples**: May require specific data structures

The examples use placeholder paths by default. Update these paths to point to your actual test files.

### Model Downloads

Some detectors download models on first use:

- `PerplexitySignal`: ~500MB (GPT-2)
- `BinocularsSignal`: ~7GB (GPT-2 + Falcon)
- `Wav2VecSignal`: ~1.2GB
- `DIRESignal`: ~4GB (Stable Diffusion)

Set cache directory if needed:
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### GPU Recommendations

- `DIRESignal` (image): GPU highly recommended
- `Wav2VecSignal` (audio): GPU recommended
- Other detectors work fine on CPU

## Documentation Sync

These examples are extracted from the documentation to ensure they stay runnable and testable. When the documentation is updated, these examples should be updated accordingly to maintain sync.

**Source Documentation:**
- `docs/tutorials/quick_start.md`
- `docs/tutorials/text_detection_guide.md`
- `docs/tutorials/image_detection_guide.md`
- `docs/tutorials/audio_detection_guide.md`
- `docs/tutorials/ensemble_detection.md`
- `docs/use_cases.md`

## Contributing

When adding new examples to documentation:

1. Extract the example to the appropriate file in this directory
2. Ensure it's runnable and well-documented
3. Add test coverage if applicable
4. Update this README if adding new files

## Questions or Issues?

See the main [README](../../README.md) or [documentation](https://adityamahakali.github.io/veridex/) for more information.
