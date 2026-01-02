
import torch
import unittest
from unittest.mock import MagicMock, patch
from veridex.text.tdetect import TDetectSignal

class TestTDetect(unittest.TestCase):
    def setUp(self):
        self.signal = TDetectSignal(
            base_model_name="gpt2",
            perturbation_model_name="google/flan-t5-small",
            n_perturbations=5
        )

    def test_initialization(self):
        self.assertEqual(self.signal.name, "t_detect")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    def test_run_mocked(self, mock_seq2seq, mock_tokenizer, mock_causal):
        # Mock base model
        mock_base_model = MagicMock()
        mock_base_model.to.return_value = mock_base_model # Handle .to(device)
        mock_base_output = MagicMock()
        mock_base_output.loss.item.return_value = 2.0
        mock_base_model.return_value = mock_base_output
        mock_causal.from_pretrained.return_value = mock_base_model

        # Mock perturbation model
        mock_perturb_model = MagicMock()
        mock_perturb_model.to.return_value = mock_perturb_model # Handle .to(device)
        mock_perturb_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_seq2seq.from_pretrained.return_value = mock_perturb_model

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_ids = MagicMock()
        mock_ids.to.return_value = mock_ids # chainable .to()
        mock_tok_output = MagicMock()
        mock_tok_output.input_ids = mock_ids
        mock_tok.side_effect = lambda x, return_tensors=None: mock_tok_output
        mock_tok.decode.return_value = "perturbed"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        result = self.signal.run("Test input")

        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("t_score", result.metadata)
        self.assertIn("df", result.metadata)

if __name__ == "__main__":
    unittest.main()
