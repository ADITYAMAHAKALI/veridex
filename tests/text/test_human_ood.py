
import torch
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from veridex.text.human_ood import HumanOODSignal

class TestHumanOOD(unittest.TestCase):
    def setUp(self):
        self.signal = HumanOODSignal(
            model_name="gpt2",
            n_samples=2, # Small n for test
            max_length=10
        )

    def test_initialization(self):
        self.assertEqual(self.signal.name, "human_ood")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_run_mocked(self, mock_tokenizer, mock_causal):
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        # Hidden states mocking
        # Shape: (1, seq_len, dim=10)
        # We need `outputs.hidden_states[-1]`
        mock_output = MagicMock()
        # Mock hidden states as tuple (one tensor)
        # Return random embedding for input and generations

        # We need dynamic return values because we call it for input and N samples
        def model_forward(*args, **kwargs):
            mock_out = MagicMock()
            # Random embedding (1, 3, 4) to match attention mask seq len
            mock_out.hidden_states = (None, None, torch.randn(1, 3, 4))
            return mock_out

        mock_model.side_effect = model_forward

        # Generation mock
        mock_model.generate.return_value = torch.tensor([[101, 102]])

        mock_causal.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "eos"

        # Encode return
        mock_tok.encode.return_value = [1, 2, 3, 4, 5, 6]

        # Call return (for embedding)
        mock_tok_out = MagicMock()
        mock_tok_out.input_ids = torch.tensor([[1, 2, 3]])
        mock_tok_out.attention_mask = torch.tensor([[1, 1, 1]])
        # Fix: .to() support
        mock_tok_out.to.return_value = mock_tok_out

        mock_tok.side_effect = lambda x, **kwargs: mock_tok_out
        mock_tok.decode.return_value = "generated text"

        mock_tokenizer.from_pretrained.return_value = mock_tok

        result = self.signal.run("This is a test input.")

        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("distance", result.metadata)
        self.assertEqual(result.metadata["n_samples"], 2)

if __name__ == "__main__":
    unittest.main()
