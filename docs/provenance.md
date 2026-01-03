# C2PA Provenance Integration

Veridex integrates the [Coalition for Content Provenance and Authenticity (C2PA)](https://c2pa.org/) standard to verify the origin of digital content. This feature allows you to detect if a file has a cryptographic signature indicating it was generated or modified by AI tools.

## Installation

The C2PA functionality requires the `c2pa-python` library. You can install it as an optional dependency:

```bash
pip install veridex[c2pa]
```

## Usage

Veridex provides specific provenance checkers for Image, Audio, and Text. These classes inherit from the core `C2PASignal` and are designed to simplify integration into your workflows.

### Image Provenance

For images (JPEG, PNG, WEBP, AVIF, HEIC), C2PA manifests are typically embedded directly in the file.

```python
from veridex.image import C2PAImageProvenance

detector = C2PAImageProvenance()
result = detector.run("path/to/image.jpg")

if result.score == 1.0:
    print("Content is signed as AI-Generated.")
elif result.metadata.get("manifest_found"):
    print("Content has a valid C2PA manifest (Human/Edited).")
else:
    print("No C2PA manifest found.")
```

### Audio Provenance

For audio (MP3, WAV, M4A), manifests are also embedded.

```python
from veridex.audio import C2PAAudioProvenance

detector = C2PAAudioProvenance()
result = detector.run("path/to/audio.mp3")
```

### Text Provenance (Sidecar Manifests)

Text files (`.txt`, `.md`, `.csv`) cannot embed binary metadata. Therefore, C2PA uses "sidecar" manifest files (typically with a `.c2pa` extension).

`C2PATextProvenance` automatically checks for a sidecar file in the same directory as the input file.

*   Input: `document.txt`
*   Checks for: `document.txt.c2pa` OR `document.c2pa`

```python
from veridex.text import C2PATextProvenance

detector = C2PATextProvenance()
# Ensure document.txt.c2pa exists in the same folder
result = detector.run("path/to/document.txt")

if result.metadata.get("sidecar_used"):
    print("Verified using sidecar manifest.")
```

## How It Works

The detector parses the C2PA manifest and looks for specific assertions:

1.  **IPTC Digital Source Type**: Checks for `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`.
2.  **C2PA Actions**: Checks if any action in the history is flagged as `trainedAlgorithmicMedia`.

If either is found, the `score` is returned as **1.0** (AI). If a manifest exists but no AI assertions are present, the score is **0.0** (Human/Natural).

## API Reference

### `C2PASignal` (Core)

Located in `veridex.core.provenance`.

*   **run(file_path: str) -> DetectionResult**
    *   Returns `DetectionResult` with:
        *   `score`: 1.0 (AI) or 0.0 (Human/Unknown)
        *   `confidence`: 1.0 (if signed)
        *   `metadata`:
            *   `manifest_found` (bool)
            *   `is_ai_signed` (bool)
            *   `found_assertions` (list)
            *   `sidecar_used` (bool)
