import unittest
import numpy as np
from PIL import Image
import os
import sys
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from veridex.image.frequency import FrequencySignal
from veridex.core.signal import DetectionResult

class TestFrequencySignal(unittest.TestCase):
    def setUp(self):
        self.signal = FrequencySignal()
        # Create a dummy image
        self.img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.pil_image = Image.fromarray(self.img_array)

    def test_run_numpy(self):
        result = self.signal.run(self.img_array)
        self.assertIsInstance(result, DetectionResult)
        self.assertIn("mean_magnitude", result.metadata)
        self.assertIn("high_freq_ratio", result.metadata)

    def test_run_pil(self):
        result = self.signal.run(self.pil_image)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.metadata["image_shape"], (100, 100))

    def test_high_freq_ratio_diff(self):
        # Create a smooth image (low freq)
        smooth = np.zeros((100, 100), dtype=np.uint8)
        # Create a noise image (high freq)
        noise = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        res_smooth = self.signal.run(smooth)
        res_noise = self.signal.run(noise)

        # Noise should have higher high_freq_ratio
        self.assertGreater(res_noise.metadata["high_freq_ratio"], res_smooth.metadata["high_freq_ratio"])


class TestDIRESignal(unittest.TestCase):
    def test_dire_run(self):
        # Since imports are inside methods, we can't easily patch the classes directly
        # unless we ensure the module uses the mocked modules.
        # But we are in the same process.

        # Strategy: Mock `importlib.import_module` or similar? No, too complex.
        # Strategy: Create a subclass that mocks _load_pipeline.

        from veridex.image.dire import DIRESignal

        class MockDIRESignal(DIRESignal):
            def check_dependencies(self):
                pass

            def _load_pipeline(self):
                # Return a mock pipeline
                mock_pipeline = MagicMock()
                mock_output = MagicMock()
                # Return black image
                mock_output.images = [Image.new("RGB", (512, 512), color="black")]
                mock_pipeline.return_value = mock_output
                return mock_pipeline

        signal = MockDIRESignal(device="cpu")

        # Run on a white image
        input_img = Image.new("RGB", (512, 512), color="white")

        result = signal.run(input_img)

        if result.error:
            print(f"Test failed with error: {result.error}")

        self.assertIsInstance(result, DetectionResult)
        self.assertIn("dire_mae", result.metadata)
        self.assertAlmostEqual(result.metadata["dire_mae"], 1.0, delta=0.1)

    def test_dire_initialization(self):
        """Test DIRE signal initialization."""
        from veridex.image.dire import DIRESignal
        
        signal = DIRESignal(device="cpu")
        self.assertEqual(signal.device, "cpu")

    def test_dire_properties(self):
        """Test DIRE signal properties."""
        from veridex.image.dire import DIRESignal
        
        signal = DIRESignal()
        self.assertEqual(signal.name, "dire_reconstruction")
        self.assertEqual(signal.dtype, "image")

    def test_dire_numpy_array_input(self):
        """Test DIRE with numpy array input."""
        from veridex.image.dire import DIRESignal
        
        class MockDIRESignal(DIRESignal):
            def check_dependencies(self):
                pass
            
            def _load_pipeline(self):
                mock_pipeline = MagicMock()
                mock_output = MagicMock()
                mock_output.images = [Image.new("RGB", (512, 512), color="gray")]
                mock_pipeline.return_value = mock_output
                return mock_pipeline
        
        signal = MockDIRESignal(device="cpu")
        img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        result = signal.run(img_array)
        self.assertIsInstance(result, DetectionResult)

    def test_dire_small_image(self):
        """Test DIRE with smaller image."""
        from veridex.image.dire import DIRESignal
        
        class MockDIRESignal(DIRESignal):
            def check_dependencies(self):
                pass
            
            def _load_pipeline(self):
                mock_pipeline = MagicMock()
                mock_output = MagicMock()
                mock_output.images = [Image.new("RGB", (256, 256), color="red")]
                mock_pipeline.return_value = mock_output
                return mock_pipeline
        
        signal = MockDIRESignal(device="cpu")
        input_img = Image.new("RGB", (256, 256), color="blue")
        
        result = signal.run(input_img)
        self.assertIsInstance(result, DetectionResult)

    def test_dire_score_bounds(self):
        """Test that DIRE score stays within bounds."""
        from veridex.image.dire import DIRESignal
        
        class MockDIRESignal(DIRESignal):
            def check_dependencies(self):
                pass
            
            def _load_pipeline(self):
                mock_pipeline =MagicMock()
                mock_output = MagicMock()
                mock_output.images = [Image.new("RGB", (512, 512), color="black")]
                mock_pipeline.return_value = mock_output
                return mock_pipeline
        
        signal = MockDIRESignal(device="cpu")
        input_img = Image.new("RGB", (512, 512), color="white")
        
        result = signal.run(input_img)
        self.assertTrue(0.0 <= result.score <= 1.0)
        self.assertTrue(0.0 <= result.confidence <= 1.0)

if __name__ == "__main__":
    unittest.main()
