from typing import Any
from veridex.core.signal import BaseSignal, DetectionResult

class DummyTextSignal(BaseSignal):
    """
    A simple dummy signal for testing purposes.
    """

    @property
    def name(self) -> str:
        return "dummy_text_signal"

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

        # Fake heuristic: if "AI" is in text, high score.
        score = 0.9 if "AI" in input_data else 0.1
        return DetectionResult(
            score=score,
            confidence=1.0,
            metadata={"length": len(input_data)}
        )
