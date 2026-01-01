import zlib
from typing import Any
from veridex.core.signal import BaseSignal, DetectionResult

class ZlibEntropySignal(BaseSignal):
    """
    Detects AI content using compression ratio (zlib entropy).

    This method employs a compression-based approach under the hypothesis that AI-generated
    content is more predictable map (lower entropy) and thus more compressible than human content.

    Algorithm:
        ratio = len(zlib(text)) / len(text)
        - Lower ratio (< 0.6) -> Highly compressible -> Likely AI.
        - Higher ratio (> 0.8) -> Less compressible -> Likely Human.

    Attributes:
        name (str): 'zlib_entropy'
        dtype (str): 'text'
    """

    @property
    def name(self) -> str:
        return "zlib_entropy"

    @property
    def dtype(self) -> str:
        return "text"

    def run(self, input_data: Any) -> DetectionResult:
        if not isinstance(input_data, str):
             return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={},
                error="Input must be a string."
            )

        if not input_data:
             return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={"zlib_ratio": 0.0},
                error="Input string is empty."
            )

        encoded = input_data.encode("utf-8")
        compressed = zlib.compress(encoded)
        ratio = len(compressed) / len(encoded)
        
        # Calculate confidence based on how extreme the ratio is
        # Very compressible (low ratio) or very incompressible (high ratio) = higher confidence
        # Middle values = lower confidence
        # Typical ranges: AI text ~0.55-0.70, Human text ~0.65-0.85
        if ratio < 0.6:
            # Very compressible (repetitive) - moderate confidence it's AI
            confidence = 0.4
            score = 0.6  # Slightly AI-leaning
        elif ratio > 0.8:
            # Not very compressible (diverse) - moderate confidence it's human
            confidence = 0.4
            score = 0.3  # Slightly human-leaning
        else:
            # Middle range - low confidence
            confidence = 0.2
            score = 0.5  # Neutral
        
        return DetectionResult(
            score=score,
            confidence=confidence,
            metadata={
                "zlib_ratio": ratio,
                "original_length": len(encoded),
                "compressed_length": len(compressed)
            }
        )
