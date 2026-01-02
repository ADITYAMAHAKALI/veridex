
import unittest
import torch
from unittest.mock import MagicMock, patch
from veridex.text.detectgpt import DetectGPTSignal

class TestDetectGPT(unittest.TestCase):
    def setUp(self):
        self.signal = DetectGPTSignal(
            base_model_name="gpt2", # smaller for test
            perturbation_model_name="google/flan-t5-small",
            n_perturbations=2
        )

    def test_initialization(self):
        self.assertEqual(self.signal.name, "detectgpt")
        self.assertEqual(self.signal.dtype, "text")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    def test_run_mocked(self, mock_seq2seq, mock_tokenizer, mock_causal):
        # Mock base model
        mock_base_model = MagicMock()
        mock_base_model.to.return_value = mock_base_model
        mock_base_output = MagicMock()
        mock_base_output.loss.item.return_value = 2.5 # log loss
        mock_base_model.return_value = mock_base_output
        mock_causal.from_pretrained.return_value = mock_base_model

        # Mock perturbation model
        mock_perturb_model = MagicMock()
        mock_perturb_model.to.return_value = mock_perturb_model
        # Mock generate output
        # Return a tensor of shape (1, 3)
        mock_perturb_model.generate.return_value = torch.tensor([[101, 200, 102]])
        mock_seq2seq.from_pretrained.return_value = mock_perturb_model

        # Mock tokenizers
        mock_tok = MagicMock()
        # Mocking input_ids for tokenizer call
        # When tokenizer is called, it returns an object with input_ids
        mock_tok_output = MagicMock()
        mock_tok_output.input_ids = torch.tensor([[1]])
        # Make the tokenizer callable
        mock_tok.side_effect = lambda x, return_tensors=None: mock_tok_output
        # Also used for decoding
        mock_tok.decode.return_value = "perturbed text"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Run
        result = self.signal.run("This is a test.")

        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("curvature", result.metadata)

    def test_empty_input(self):
        result = self.signal.run("")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.error, "Invalid input")

if __name__ == "__main__":
    unittest.main()
