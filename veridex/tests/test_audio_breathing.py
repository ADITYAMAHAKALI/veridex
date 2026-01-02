
import unittest
import numpy as np
import tempfile
import os
import soundfile as sf
from veridex.audio.breathing_signal import BreathingSignal
from veridex.core.signal import DetectionResult

class TestBreathingSignal(unittest.TestCase):
    def setUp(self):
        self.signal = BreathingSignal()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_mock_audio(self, duration=10.0, sr=16000, type="silence"):
        t = np.linspace(0, duration, int(sr * duration))

        if type == "silence":
            audio = np.zeros_like(t)
        elif type == "sine":
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        elif type == "noise":
            audio = 0.1 * np.random.normal(0, 1, len(t))
        elif type == "breath_mock":
            # Speech-like (sine) with Noise bursts (breath) and Silence
            # We need Silence < Breath < Speech for the energy heuristic to work
            audio = 0.5 * np.sin(2 * np.pi * 440 * t) # continuous tone (Speech)

            # Add silence at the beginning
            audio[:int(0.5*sr)] = 0.0

            # Add "breaths" every 3 seconds
            for start_t in range(1, int(duration), 3):
                start_idx = int(start_t * sr)
                end_idx = int((start_t + 0.5) * sr)
                if end_idx < len(audio):
                    # Silence the tone first
                    audio[start_idx:end_idx] = 0
                    # Add noise (breath) - approx 0.1 amplitude (Speech is 0.5)
                    # High frequency noise
                    noise = np.random.normal(0, 0.05, end_idx - start_idx)
                    # Simple high pass filter simulation by differencing
                    noise = np.diff(noise, append=0)
                    audio[start_idx:end_idx] = noise
        else:
            audio = np.zeros_like(t)

        filepath = os.path.join(self.temp_dir, f"test_{type}.wav")
        sf.write(filepath, audio, sr)
        return filepath

    def test_initialization(self):
        self.assertEqual(self.signal.name, "breathing_audio_detector")
        self.assertEqual(self.signal.dtype, "audio")

    def test_run_invalid_input(self):
        result = self.signal.run(123)
        self.assertIsInstance(result, DetectionResult)
        self.assertTrue(result.error)

    def test_run_silence(self):
        # Silence -> Validated by utils as "Audio signal is silent" -> Error
        filepath = self.create_mock_audio(type="silence", duration=5.0)
        result = self.signal.run(filepath)

        # The validate_audio utility should catch silence and return an error
        self.assertIsNotNone(result.error)
        self.assertIn("silent", result.error)

    def test_run_mock_breaths(self):
        # Should detect breaths
        filepath = self.create_mock_audio(type="breath_mock", duration=10.0)
        result = self.signal.run(filepath)

        self.assertIsNone(result.error)
        # We inserted breaths every 3s in 10s audio -> ~3 breaths.
        # BPM ~ 18. This is human-like.
        # Score should be low (Human).

        # Check metadata
        self.assertIn("breaths_per_minute", result.metadata)
        bpm = result.metadata["breaths_per_minute"]
        self.assertGreater(bpm, 5.0)

        # Score should be low (likely human)
        self.assertLess(result.score, 0.5)

if __name__ == "__main__":
    unittest.main()
