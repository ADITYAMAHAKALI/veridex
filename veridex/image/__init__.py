"""
Image detection module for AI-generated image content.

Provides detectors for identifying synthetic images:
- FrequencySignal: Frequency domain analysis (lightweight)
- DIRESignal: Diffusion reconstruction error (high accuracy)
"""

from .frequency import FrequencySignal
from .dire import DIRESignal

__all__ = [
    "FrequencySignal",
    "DIRESignal",
]
