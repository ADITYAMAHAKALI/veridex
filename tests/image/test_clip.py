import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch # Import torch at top level to avoid reload issues
from veridex.image.clip import CLIPSignal

class TestCLIPSignal(unittest.TestCase):
    def setUp(self):
        self.signal = CLIPSignal(device="cpu")

    def test_initialization(self):
        self.assertEqual(self.signal.name, "clip_zeroshot")
        self.assertEqual(self.signal.dtype, "image")

    @patch("veridex.image.clip.CLIPSignal._load_model")
    def test_run_mocked(self, mock_load):
        # Mock the loaded model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_load.return_value = (mock_model, mock_processor)

        # Mock inference output
        num_prompts = len(self.signal.real_prompts) + len(self.signal.fake_prompts)

        # Create dummy logits
        logits = torch.zeros((1, num_prompts))
        # Set fake prompt indices to high value
        logits[0, -1] = 10.0

        mock_output = MagicMock()
        mock_output.logits_per_image = logits
        mock_model.return_value = mock_output

        # Mock processor
        mock_processor.return_value = {"input_ids": torch.tensor([[1]]), "pixel_values": torch.tensor([[[[1.0]]]])}
        # Ensure 'to' returns the dict itself (or a mocked object behaving like it)
        # In reality processor returns a BatchEncoding which has .to()
        # We can just return the dict and mock .to call on the result of processor() if needed.
        # But wait, in code: inputs = processor(...).to(self.device)
        # So processor(...) must return an object with .to() method.

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # Create dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Run
        result = self.signal.run(dummy_image)

        # Assertions
        self.assertGreater(result.score, 0.5)
        self.assertIn("top_prompt", result.metadata)
        self.assertIn("prob_fake", result.metadata)

    def test_missing_dependencies(self):
        # We need to simulate missing transformers.
        # Since we imported torch/transformers at top level, we need to be careful.
        # patch.dict should work for the scope of this test.
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError):
                self.signal.check_dependencies()
