from typing import Any, Optional
import math
import numpy as np
from veridex.core.signal import BaseSignal, DetectionResult

class PerplexitySignal(BaseSignal):
    """
    Analyzes text complexity using Perplexity metrics.

    This signal assumes that AI-generated text tends to have lower perplexity (more predictable)
    compared to human-written text.

    **Mechanism**:
    1.  Tokenize input text using a pretrained tokenizer (e.g., GPT-2).
    2.  Calculate perplexity using the corresponding language model.
    3.  Map perplexity to an AI probability score using a logistic function.
        - Low perplexity -> High AI Probability.
        - High perplexity -> Low AI Probability (Human).

    Attributes:
        model_name (str): Name of the underlying model (default: "gpt2").
        device (Optional[str]): Device to run model on ('cpu', 'cuda').
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the Perplexity signal.

        Args:
            model_name (str): Identifier for the model used to calculate perplexity.
            device (Optional[str]): Device to run model on.
        """
        self.model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        """Returns 'perplexity'."""
        return "perplexity"

    @property
    def dtype(self) -> str:
        """Returns 'text'."""
        return "text"

    def check_dependencies(self) -> None:
        try:
            import torch
            import transformers
        except ImportError:
            raise ImportError(
                "PerplexitySignal requires 'torch' and 'transformers'. "
                "Install with `pip install veridex[text]`"
            )

    def _load_model(self):
        if self._model is not None:
            return

        self.check_dependencies()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def run(self, input_data: Any) -> DetectionResult:
        """
        Calculate perplexity and convert to an AI score.

        Args:
            input_data (str): Text to analyze.

        Returns:
            DetectionResult:
                - score: 0.0-1.0 AI probability.
                - metadata: {'mean_perplexity', 'model_id'}.
        """
        if not isinstance(input_data, str):
             return DetectionResult(score=0.0, confidence=0.0, error="Input must be a string")

        try:
            self._load_model()
            import torch

            # Tokenize with truncation to max length (usually 1024 for GPT-2)
            # This prevents crashes on long text.
            inputs = self._tokenizer(
                input_data,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            if inputs["input_ids"].shape[1] < 2:
                 # Too short for meaningful perplexity
                 return DetectionResult(score=0.5, confidence=0.0, metadata={"reason": "Text too short"})

            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()

            # Heuristic mapping from Perplexity to AI Score
            # Threshold = 50. If PPL < 50, P(AI) > 0.5.
            threshold = 50.0
            scale = 10.0
            score = 1.0 / (1.0 + np.exp((perplexity - threshold) / scale))

            return DetectionResult(
                score=float(score),
                confidence=0.7,
                metadata={
                    "mean_perplexity": perplexity,
                    "model_id": self.model_name
                }
            )

        except ImportError as e:
            return DetectionResult(score=0.0, confidence=0.0, error=str(e))
        except Exception as e:
             return DetectionResult(score=0.0, confidence=0.0, error=str(e))
