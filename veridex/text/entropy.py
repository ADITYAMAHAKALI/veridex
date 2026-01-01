import zlib
from typing import Any
from veridex.core.signal import BaseSignal, DetectionResult

class ZlibEntropySignal(BaseSignal):
    """
    Calculates the zlib compression ratio of the text.
    Lower ratio means the text is more compressible (repetitive, low entropy).

    This signal primarily provides the compression ratio in metadata.
    The 'score' is set to 0.5 (neutral) as this metric alone is insufficient
    for classification without a reference distribution or calibration.
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

        return DetectionResult(
            score=0.5,
            confidence=0.0,
            metadata={
                "zlib_ratio": ratio,
                "original_length": len(encoded),
                "compressed_length": len(compressed)
            }
        )
