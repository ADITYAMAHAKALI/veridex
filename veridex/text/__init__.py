"""
Text detection module for AI-generated text content.

Provides detectors for identifying synthetic text:
- ZlibEntropySignal: Compression-based entropy analysis
- PerplexitySignal: LLM-based perplexity analysis
"""

from .entropy import ZlibEntropySignal
from .perplexity import PerplexitySignal

__all__ = [ "ZlibEntropySignal", "PerplexitySignal"]
