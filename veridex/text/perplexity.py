from typing import Any
from veridex.core.signal import BaseSignal, DetectionResult

class PerplexitySignal(BaseSignal):
    """
    Calculates the perplexity of the text using a pre-trained language model (e.g. GPT-2).
    Requires 'transformers' and 'torch' to be installed.
    """

    def __init__(self, model_id: str = "gpt2"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def dtype(self) -> str:
        return "text"

    def check_dependencies(self) -> None:
        try:
            import torch
            import transformers
        except ImportError:
            raise ImportError(
                "The 'text' extra dependencies (transformers, torch) are required for PerplexitySignal. "
                "Install them with `pip install veridex[text]`."
            )

    def _load_model(self):
        if self._model is not None:
            return

        self.check_dependencies()

        # Local import to avoid top-level dependency
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Use CPU by default for broader compatibility in this context
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id).to(device)

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
                metadata={},
                error="Input string is empty."
            )

        try:
            self._load_model()
        except ImportError as e:
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={},
                error=str(e)
            )
        except Exception as e:
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={},
                error=f"Failed to load model '{self.model_id}': {e}"
            )

        import torch

        try:
            inputs = self._tokenizer(input_data, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()

            return DetectionResult(
                score=0.5, # Again, raw metric without calibration
                confidence=0.0,
                metadata={
                    "perplexity": perplexity,
                    "model_id": self.model_id
                }
            )

        except Exception as e:
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={},
                error=f"Error calculating perplexity: {e}"
            )
