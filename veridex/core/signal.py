from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class DetectionResult(BaseModel):
    """
    Standardized output for all detection signals.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score indicating AI probability. 0=Human, 1=AI.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Reliability of the score estimation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal-specific information.")
    error: Optional[str] = Field(None, description="Error message if the signal failed to execute.")

class BaseSignal(ABC):
    """
    Abstract base class for all detection signals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the signal."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> str:
        """Data type this signal operates on (e.g., 'text', 'image')."""
        pass

    @abstractmethod
    def run(self, input_data: Any) -> DetectionResult:
        """
        Execute the detection logic.
        """
        pass

    def check_dependencies(self) -> None:
        """
        Optional hook to check if required heavy dependencies are installed.
        Should raise ImportError if missing.
        """
        pass
