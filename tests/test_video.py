import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np
import os

# NOTE: All global mocks (torch, torchvision, cv2, av, scipy, librosa) are handled in conftest.py
# This ensures consistent mocking across all test files and prevents import issues

# Import signals (will need to import them inside test or after mocks if they rely on top-level imports)
# Since we are implementing them, we will ensure they use lazy imports or handle top-level imports gracefully.
# For now, we assume the classes will be available in veridex.video

class TestVideoSignals(unittest.TestCase):

    def setUp(self):
        # Create a dummy video file
        self.dummy_video_path = "dummy_video.mp4"
        with open(self.dummy_video_path, "w") as f:
            f.write("dummy content")

    def tearDown(self):
        if os.path.exists(self.dummy_video_path):
            os.remove(self.dummy_video_path)

    def test_rppg_signal_structure(self):
        """Test the RPPGSignal interface and logic flow."""
        # We need to import inside to ensure mocks are active if we had real imports
        from veridex.video.rppg import RPPGSignal

        signal = RPPGSignal()
        self.assertEqual(signal.name, "rppg_physnet")
        self.assertEqual(signal.dtype, "video")

        # Mock the run methods
        with patch.object(signal, '_load_video_frames') as mock_load, \
             patch.object(signal, '_detect_faces') as mock_face, \
             patch.object(signal, '_extract_signal') as mock_extract, \
             patch.object(signal, '_analyze_psd') as mock_psd:

            mock_load.return_value = np.zeros((30, 128, 128, 3)) # 30 frames
            mock_face.return_value = [np.zeros((30, 64, 64, 3))] # 1 face track
            # Updated: _extract_signal now returns (signal, weights_loaded)
            mock_extract.return_value = (np.sin(np.linspace(0, 10, 30)), False) # Signal + untrained

            # Case 1: Fake
            # Return tuple (score, metadata)
            mock_psd.return_value = (0.8, {"snr": 0.1, "peak_ratio": 1.0})

            result = signal.run(self.dummy_video_path)
            # Check if result.error is set, which would indicate a crash
            if result.error:
                print(f"RPPG Error: {result.error}")

            self.assertEqual(result.score, 0.8)
            self.assertIn('snr', result.metadata)

    def test_i3d_signal_structure(self):
        """Test the I3DSignal interface."""
        from veridex.video.i3d import I3DSignal

        signal = I3DSignal()
        self.assertEqual(signal.name, "spatiotemporal_i3d")
        self.assertEqual(signal.dtype, "video")

        with patch.object(signal, '_load_clip') as mock_load, \
             patch.object(signal, '_run_inference') as mock_inf:

            mock_load.return_value = np.zeros((64, 224, 224, 3))
            # Updated: _run_inference now returns (score, weights_loaded)
            mock_inf.return_value = (0.95, True) # Confidently AI, trained model

            result = signal.run(self.dummy_video_path)
            if result.error:
                print(f"I3D Error: {result.error}")

            self.assertEqual(result.score, 0.95)
            # Confidence is dynamic now, just check it exists and is reasonable
            self.assertGreater(result.confidence, 0.7)  # Should be high for trained model

    def test_lipsync_signal_structure(self):
        """Test the LipSyncSignal interface."""
        from veridex.video.lipsync import LipSyncSignal

        signal = LipSyncSignal()
        self.assertEqual(signal.name, "lipsync_wav2lip")
        self.assertEqual(signal.dtype, "video")

        with patch.object(signal, '_calculate_av_offset') as mock_offset:
            # Updated: _calculate_av_offset now returns (offset, weights_loaded)
            mock_offset.return_value = (2.0, True)  # High offset, trained model
            # In code: if offset > 0.8 -> score = (2.0 - 0.8) / 1.0 = 1.0 (capped)

            result = signal.run(self.dummy_video_path)
            if result.error:
                print(f"LipSync Error: {result.error}")

            self.assertEqual(result.score, 1.0)
            self.assertIn("av_distance", result.metadata)

if __name__ == '__main__':
    unittest.main() 
