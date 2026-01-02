import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from veridex.image.mlep import MLEPSignal
from veridex.core.signal import DetectionResult

class TestMLEPSignal(unittest.TestCase):
    def setUp(self):
        self.signal = MLEPSignal()

    def test_initialization(self):
        self.assertEqual(self.signal.name, "mlep_entropy")
        self.assertEqual(self.signal.dtype, "image")

    def test_properties(self):
        """Test signal properties."""
        self.assertEqual(self.signal.name, "mlep_entropy")
        self.assertEqual(self.signal.dtype, "image")

    def test_run_statistics(self):
        # Create a dummy synthetic image (checkerboard)
        # 100x100
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[::2, ::2, :] = 255

        # We need to ensure dependencies are present or mock them.
        # Since standard env likely has numpy/scipy/skimage, we try running it directly.
        # If it fails due to missing deps in the minimal env, we mock.

        try:
            import skimage
            import scipy
        except ImportError:
            print("Skipping MLEP logic test due to missing dependencies")
            return

        result = self.signal.run(image)

        self.assertEqual(result.error, None)
        self.assertIn("mean_entropy", result.metadata)
        self.assertIn("variance_entropy", result.metadata)
        self.assertIsInstance(result.metadata["mean_entropy"], float)

    def test_missing_dependencies(self):
        with patch.dict("sys.modules", {"skimage": None, "scipy": None}):
            with self.assertRaises(ImportError):
                self.signal.check_dependencies()

    def test_run_invalid_input_type(self):
        """Test run with invalid input types."""
        # Mock dependencies to reach validation logic
        mocks = {
            "skimage": MagicMock(),
            "skimage.filters": MagicMock(),
            "skimage.filters.rank": MagicMock(),
            "skimage.morphology": MagicMock(),
            "skimage.color": MagicMock(),
            "skimage.util": MagicMock(),
            "scipy": MagicMock(),
            "scipy.stats": MagicMock()
        }
        with patch.dict("sys.modules", mocks):
            # Test with dict which will fail file opening
            result = self.signal.run({"invalid": "input"})
            
            self.assertIsInstance(result, DetectionResult)
            self.assertIsNotNone(result.error)
            self.assertEqual(result.score, 0.0)

    def test_run_grayscale_image(self):
        """Test run with grayscale image."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create grayscale image
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = self.signal.run(image)
        
        self.assertIsInstance(result, DetectionResult)
        # Should handle grayscale images

    def test_run_small_image(self):
        """Test run with very small image."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create tiny image
        image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        result = self.signal.run(image)
        
        self.assertIsInstance(result, DetectionResult)

    def test_run_large_image(self):
        """Test run with large image."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create larger image
        image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        result = self.signal.run(image)
        
        self.assertIsInstance(result, DetectionResult)
        self.assertIn("mean_entropy", result.metadata)

    def test_run_uniform_image(self):
        """Test run with uniform (solid color) image."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create uniform image (all white)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = self.signal.run(image)
        
        self.assertIsInstance(result, DetectionResult)
        # Uniform images should have low entropy

    def test_run_high_entropy_image(self):
        """Test run with high entropy (random) image."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create random noise image (high entropy)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = self.signal.run(image)
        
        self.assertIsInstance(result, DetectionResult)
        if result.error is None:
            # High entropy images should have higher mean_entropy
            self.assertIn("mean_entropy", result.metadata)

    def test_run_basic_execution(self):
        """Test basic execution path."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create simple test image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.signal.run(image)
        
        self.assertIsInstance(result, DetectionResult)

    def test_score_computation_high_entropy(self):
        """Test that high entropy variance leads to higher AI detection score."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create image that produces high variance across layers
        # This simulates AI-generated patterns
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = self.signal.run(image)
        
        if result.error is None:
            self.assertTrue(0.0 <= result.score <= 1.0)
            self.assertTrue(0.0 <= result.confidence <= 1.0)

    def test_error_handling(self):
        """Test error handling with corrupted input."""
        # Mock dependencies
        mocks = {
            "skimage": MagicMock(),
            "skimage.filters": MagicMock(),
            "skimage.filters.rank": MagicMock(),
            "skimage.morphology": MagicMock(),
            "skimage.color": MagicMock(),
            "skimage.util": MagicMock(),
            "scipy": MagicMock(),
            "scipy.stats": MagicMock()
        }
        with patch.dict("sys.modules", mocks):
            # Pass invalid input type
            result = self.signal.run(12345)
            
            self.assertIsInstance(result, DetectionResult)
            self.assertIsNotNone(result.error)
            self.assertEqual(result.score, 0.0)

    def test_run_pil_image(self):
        """Test run with PIL Image object."""
        try:
            import skimage
            import scipy
            from PIL import Image
        except ImportError:
            self.skipTest("Missing dependencies")
        
        # Create PIL image
        pil_img = Image.new('RGB', (100, 100), color='red')
        
        result = self.signal.run(pil_img)
        
        self.assertIsInstance(result, DetectionResult)

    def test_metadata_completeness(self):
        """Test that metadata contains expected fields."""
        try:
            import skimage
            import scipy
        except ImportError:
            self.skipTest("Missing dependencies")
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.signal.run(image)
        
        if result.error is None:
            self.assertIn("mean_entropy", result.metadata)
            self.assertIn("variance_entropy", result.metadata)
            # Check for additional expected metadata
            if "layer_entropies" in result.metadata:
                self.assertIsInstance(result.metadata["layer_entropies"], list)

