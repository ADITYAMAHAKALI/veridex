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

    # Fake file path for demonstration
    image_path = "signed_image.jpg"
    # Ensure file exists or handle error (here we just run it to show logic)
    if os.path.exists(image_path):
        result = image_prov.run(image_path)
        print(f"File: {image_path}")
        print(f"Is AI Signed: {result.score == 1.0}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"File {image_path} not found (skipping real execution).")

    # 2. Text Verification (Sidecar Manifest)
    print("\n--- Text Verification ---")
    text_prov = C2PATextProvenance()

    text_path = "document.txt"
    # Logic expects document.txt.c2pa or document.c2pa to exist
    if os.path.exists(text_path):
        result = text_prov.run(text_path)
        print(f"File: {text_path}")
        print(f"Is AI Signed: {result.score == 1.0}")
        print(f"Sidecar Used: {result.metadata.get('sidecar_used', False)}")
    else:
        print(f"File {text_path} not found (skipping real execution).")

    # 3. Audio Verification
    print("\n--- Audio Verification ---")
    audio_prov = C2PAAudioProvenance()

    audio_path = "recording.mp3"
    if os.path.exists(audio_path):
        result = audio_prov.run(audio_path)
        print(f"File: {audio_path}")
        print(f"Is AI Signed: {result.score == 1.0}")
    else:
        print(f"File {audio_path} not found (skipping real execution).")

if __name__ == "__main__":
    verify_files()
