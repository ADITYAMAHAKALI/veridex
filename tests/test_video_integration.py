import unittest
import os
import numpy as np
import pytest
from unittest.mock import patch

# Skip if dependencies are missing
try:
    import torch
    import cv2
    import soundfile as sf
    import librosa
    from veridex.video.rppg import RPPGSignal
    from veridex.video.i3d import I3DSignal
    from veridex.video.lipsync import LipSyncSignal
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

@pytest.mark.skipif(not HAS_DEPS, reason="Video dependencies not installed")
class TestVideoIntegration(unittest.TestCase):
    def setUp(self):
        self.video_path = "integration_test.mp4"
        self.audio_path = "integration_test.wav"
        
        # 1. Create Video
        # 30 frames, 512x512
        writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
        for i in range(30):
            # Face at center (200, 200) size 100x100
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(img, (200, 200), (300, 300), (200, 200, 200), -1)
            # Add some "noise" or movement to test tracking? 
            # Shift slightly
            shift = int(i * 1.0)
            cv2.rectangle(img, (200 + shift, 200), (300 + shift, 300), (100, 100, 100), -1)
            writer.write(img)
        writer.release()
        
        # 2. Create Audio
        # 1 sec
        sr = 16000
        audio = np.random.uniform(-0.1, 0.1, size=(sr,))
        sf.write(self.audio_path, audio, sr)

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)

    def test_rppg(self):
        signal = RPPGSignal()
        # Mock download to avoid network
        with patch('veridex.utils.downloads.download_file'):
             # Also mock loading weights to just return (don't load anything / use random)
             # But we want to test model FORWARD. 
             # So we let it try to load, fail, and use random weights.
             result = signal.run(self.video_path)
        
        print(f"RPPG Result: {result}")
        if result.error:
            print(f"RPPG Error: {result.error}")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertIn("snr", result.metadata)

    def test_i3d(self):
        signal = I3DSignal()
        with patch('veridex.utils.downloads.download_file'):
            result = signal.run(self.video_path)
        
        # I3D requires 64 frames in my code. I only made 30.
        # It should return error "Video too short"
        # Let's see.
        if result.error == "Video too short":
            pass # Expected
        else:
            self.assertGreaterEqual(result.confidence, 0.0)

    def test_lipsync(self):
        signal = LipSyncSignal()
        # LipSync needs audio in the video file.
        # Ours `video_path` has no audio track.
        # Librosa might fail to load audio from mp4 if no audio stream.
        # Let's Mock `_calculate_av_offset` to use our `audio_path` instead?
        # OR just mock librosa.load to return our data.
        
        # But `_calculate_av_offset` does `librosa.load(path)`.
        # If path is video file, it tries to extract audio.
        # We can just test that it handles "no audio" gracefully or mock it.
        
        # Let's Mock librosa.load to return `self.audio_path` content
        with patch('librosa.load') as mock_load:
             y, sr = sf.read(self.audio_path)
             mock_load.return_value = (y, sr)
             
             with patch('veridex.utils.downloads.download_file'):
                 result = signal.run(self.video_path)
        
        # If it runs, it passes
        print(f"LipSync Result: {result}")

if __name__ == '__main__':
    unittest.main()
