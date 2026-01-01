import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from veridex.image.mlep import MLEPSignal

class TestMLEPSignal(unittest.TestCase):
    def setUp(self):
        self.signal = MLEPSignal()

    def test_initialization(self):
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
