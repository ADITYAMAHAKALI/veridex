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

        # Mock processor to return object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # Create a simpler mock that directly returns what we need
        num_prompts = len(self.signal.real_prompts) + len(self.signal.fake_prompts)
        num_real = len(self.signal.real_prompts)
        
        # Mock the entire model output pipeline
        # We need to mock: outputs.logits_per_image.softmax(dim=1)[0, :num_real].sum().item()
        # and outputs.logits_per_image.softmax(dim=1)[0, num_real:].sum().item()
        
        # Create a custom mock class that handles the chained calls
        class MockTensor:
            def __init__(self):
                pass
            
            def softmax(self, dim):
                # Return self to allow chaining
                return self
                
            def __getitem__(self, key):
                # key will be like (0, slice(None, num_real)) or (0, slice(num_real, None))
                if isinstance(key, tuple) and len(key) == 2:
                    row, col = key
                    if isinstance(col, slice):
                        # Return a mock that handles .sum().item()
                        mock_slice = MagicMock()
                        if col.stop == num_real or (col.start is None and col.stop == num_real):
                            # Real prompts slice
                            mock_slice.sum.return_value.item.return_value = 0.1
                        elif col.start == num_real:
                            # Fake prompts slice  
                            mock_slice.sum.return_value.item.return_value = 0.9
                        return mock_slice
                # For probs[0].argmax().item()
                elif key == 0:
                    mock_row = MagicMock()
                    mock_row.argmax.return_value.item.return_value = num_prompts - 1
                    return mock_row
                return self
        
        mock_logits = MockTensor()
        mock_output = MagicMock()
        mock_output.logits_per_image = mock_logits
        mock_model.return_value = mock_output

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
