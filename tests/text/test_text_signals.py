import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib

from veridex.text.entropy import ZlibEntropySignal
from veridex.text.perplexity import PerplexitySignal

class TestTextSignals(unittest.TestCase):

    def test_zlib_entropy(self):
        signal = ZlibEntropySignal()
        text = "This is a test string. " * 10
        result = signal.run(text)

        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.score, 0.6, delta=0.01)
        self.assertIn("zlib_ratio", result.metadata)
        self.assertLess(result.metadata["zlib_ratio"], 1.0) # Should compress somewhat

        # Test empty input
        result_empty = signal.run("")
        self.assertIsNotNone(result_empty.error)

        # Test invalid input
        result_invalid = signal.run(123)
        self.assertIsNotNone(result_invalid.error)

    def test_perplexity_missing_deps(self):
        # We need to ensure 'torch' and 'transformers' are NOT in sys.modules
        # or are None for this test.
        with patch.dict(sys.modules, {'transformers': None, 'torch': None}):
             signal = PerplexitySignal()
             # Should fail check_dependencies
             with self.assertRaises(ImportError):
                 signal.check_dependencies()

             # run() should catch the ImportError and return error in result
             result = signal.run("test")
             self.assertIsNotNone(result.error)
             self.assertIn("required for PerplexitySignal", result.error)

    def test_perplexity_success_mocked(self):
        # Create mock modules
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        mock_numpy = MagicMock()

        # Setup mock behavior
        mock_torch.cuda.is_available.return_value = False
        mock_torch.exp.return_value.item.return_value = 10.5 # Perplexity value
        mock_numpy.mean.return_value = 10.5
        mock_numpy.std.return_value = 0.0

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        # Mock tokenizer call
        mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.return_value = mock_inputs

        # Mock model call
        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        # Inject mocks into sys.modules
        # We also need to mock numpy inside the module run
        with patch.dict(sys.modules, {'torch': mock_torch, 'transformers': mock_transformers}):
            with patch('numpy.mean', return_value=10.5) as mock_mean:
                with patch('numpy.std', return_value=0.0) as mock_std:
                    signal = PerplexitySignal()

                    # Now run signal.run("test")
                    # This will trigger:
                    # 1. _load_model() -> imports transformers/torch (which are our mocks)
                    # 2. uses tokenizer/model to calculate perplexity

                    result = signal.run("This is a test.")

                    self.assertIsNone(result.error, msg=f"Signal run failed with error: {result.error}")

                    # Score is no longer hardcoded to 0.5.
                    # With PPL=10.5, score should be roughly:
                    # score = 1.0 / (1.0 + exp((10.5 - 50) / 10)) = 1 / (1 + exp(-3.95)) = 1 / (1 + 0.019) ~= 0.98
                    self.assertGreater(result.score, 0.9)

                    self.assertAlmostEqual(result.metadata["mean_perplexity"], 10.5)
                    self.assertEqual(result.metadata["model_id"], "gpt2")

if __name__ == '__main__':
    unittest.main()
