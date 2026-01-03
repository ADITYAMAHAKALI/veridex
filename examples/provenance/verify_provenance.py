"""
Example: C2PA Provenance Verification using Veridex.

This script demonstrates how to verify C2PA manifests for Image, Audio, and Text files.
It assumes you have installed veridex with c2pa support: `pip install veridex[c2pa]`
"""

import os
from veridex.image import C2PAImageProvenance
from veridex.audio import C2PAAudioProvenance
from veridex.text import C2PATextProvenance

def verify_files():
    # 1. Image Verification (Embedded Manifest)
    print("--- Image Verification ---")
    image_prov = C2PAImageProvenance()

    # Real file path validation
    image_path = "samples/provenance/CACA.jpg"
    if os.path.exists(image_path):
        result = image_prov.run(image_path)
        print(f"File: {image_path}")
        print(f"Is AI Signed: {result.score == 1.0}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"File {image_path} not found.")

    # 1.5 PDF Verification (using Image class for now as it supports embedded)
    print("\n--- PDF Verification ---")
    pdf_path = "samples/provenance/basic-signed.pdf"
    if os.path.exists(pdf_path):
        result = image_prov.run(pdf_path) # C2PA works on file structure
        print(f"File: {pdf_path}")
        print(f"Is AI Signed: {result.score == 1.0}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"File {pdf_path} not found.")

    # 2. Text Verification (Sidecar Manifest)
    print("\n--- Text Verification ---")
    text_prov = C2PATextProvenance()
    # ... (existing text code)

    # 3. Audio/Video Verification
    print("\n--- Audio/Video Verification ---")
    audio_prov = C2PAAudioProvenance()

    audio_path = "samples/provenance/BigBuckBunny_320x180.mp4"
    if os.path.exists(audio_path):
        result = audio_prov.run(audio_path)
        print(f"File: {audio_path}")
        print(f"Is AI Signed: {result.score == 1.0}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"File {audio_path} not found.")

if __name__ == "__main__":
    verify_files()
