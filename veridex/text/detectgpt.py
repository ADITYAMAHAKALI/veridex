from typing import Any, Optional, List
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    F = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from veridex.core.signal import BaseSignal, DetectionResult

class DetectGPTSignal(BaseSignal):
    """
    Implements the DetectGPT zero-shot detection method.

    DetectGPT uses the curvature of the model's log-probability function to distinguish
    human-written text from AI-generated text. The core idea is that AI-generated text occupies
    regions of negative log-curvature.

    **Reference**:
    "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
    (Mitchell et al., 2023).

    **Algorithm**:
    1. Compute log-probability of original text `log p(x)`.
    2. Generate `k` perturbed versions `x_tilde` using a mask-filling model (e.g., T5).
    3. Compute log-probability of perturbations `log p(x_tilde)`.
    4. Calculate curvature score: `log p(x) - mean(log p(x_tilde))`.
    5. Normalize score to [0, 1] range.

    **Note**:
    This signal is computationally expensive as it requires loading two LLMs (Base and Perturbation)
    and running multiple forward passes.

    Attributes:
        base_model_name (str): HuggingFace model ID for probability computation.
        perturbation_model_name (str): HuggingFace model ID for perturbation (T5).
        n_perturbations (int): Number of perturbed samples to generate.
        device (str): Computation device ('cpu', 'cuda').
    """

    def __init__(self,
                 base_model_name: str = "gpt2-medium",
                 perturbation_model_name: str = "t5-base",
                 n_perturbations: int = 10,
                 device: Optional[str] = None):
        """
        Initialize the DetectGPT signal.

        Args:
            base_model_name (str): Name of the model to use for computing log-probabilities.
                Defaults to "gpt2-medium".
            perturbation_model_name (str): Name of the model to use for generating perturbations.
                Defaults to "t5-base".
            n_perturbations (int): Number of perturbed samples to generate. Defaults to 10.
            device (Optional[str]): Device to run models on ('cpu' or 'cuda'). If None,
                auto-detects CUDA.
        """
        self.base_model_name = base_model_name
        self.perturbation_model_name = perturbation_model_name
        self.n_perturbations = n_perturbations
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

        self._base_model = None
        self._base_tokenizer = None
        self._perturb_model = None
        self._perturb_tokenizer = None

    @property
    def name(self) -> str:
        """Returns 'detect_gpt'."""
        return "detect_gpt"

    @property
    def dtype(self) -> str:
        """Returns 'text'."""
        return "text"

    def _load_base_model(self):
        if self._base_model is None:
            self.check_dependencies()
            self._base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self._base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
            self._base_model.eval()

    def _load_perturb_model(self):
        # Simplification: For this implementation, we might simulate perturbations or use a simpler
        # heuristic if T5 is too heavy, but let's stick to the interface.
        # Ideally, we would load T5ForConditionalGeneration here.
        pass

    def check_dependencies(self):
        if torch is None or AutoModelForCausalLM is None:
            raise ImportError(
                "DetectGPTSignal requires 'torch' and 'transformers'. "
                "Install with `pip install veridex[text]`"
            )

    def _get_log_prob(self, text: str) -> float:
        """Computes the log probability of a text under the base model."""
        self._load_base_model()
        inputs = self._base_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._base_model(**inputs, labels=inputs["input_ids"])
            # loss is the negative log likelihood
            log_likelihood = -outputs.loss.item()
        return log_likelihood

    def _perturb_text(self, text: str) -> List[str]:
        """
        Generates perturbed versions of the text.
        For a true DetectGPT, we would use T5 mask filling.

        This implementation currently uses a random swap heuristic for demonstration/speed
        unless the full T5 stack is implemented.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of perturbed text strings.
        """
        perturbations = []
        words = text.split()
        if len(words) < 5:
            return [text] * self.n_perturbations

        import random
        for _ in range(self.n_perturbations):
            # Simple swap of two random words
            new_words = words[:]
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            perturbations.append(" ".join(new_words))

        return perturbations

    def run(self, input_data: Any) -> DetectionResult:
        """
        Analyzes the text using DetectGPT logic.

        Args:
            input_data (str): The text to analyze.

        Returns:
            DetectionResult: Result object containing the curvature-based score.
                - score: 0.0 (Human) to 1.0 (AI).
                - metadata: Contains 'curvature', 'original_log_prob'.
        """
        if not isinstance(input_data, str):
             return DetectionResult(score=0.0, confidence=0.0, error="Input must be a string", metadata={})

        try:
            original_log_prob = self._get_log_prob(input_data)
            perturbed_texts = self._perturb_text(input_data)

            perturbed_log_probs = []
            for p_text in perturbed_texts:
                perturbed_log_probs.append(self._get_log_prob(p_text))

            mean_p_log_prob = np.mean(perturbed_log_probs)
            std_p_log_prob = np.std(perturbed_log_probs) + 1e-8

            # DetectGPT score: higher means more likely generated by the model (or similar models)
            curvature = original_log_prob - mean_p_log_prob

            # Sigmoid-like scaling
            score = 1 / (1 + np.exp(-curvature))

            # Confidence based on variance of perturbations?
            confidence = 0.5 # Placeholder

            return DetectionResult(
                score=float(score),
                confidence=confidence,
                metadata={
                    "curvature": curvature,
                    "original_log_prob": original_log_prob,
                    "mean_perturbed_log_prob": mean_p_log_prob
                }
            )

        except Exception as e:
            return DetectionResult(score=0.0, confidence=0.0, error=str(e), metadata={})
