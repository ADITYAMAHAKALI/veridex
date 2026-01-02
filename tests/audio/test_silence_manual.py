import numpy as np
import librosa
from veridex.audio import SilenceSignal

def test_silence():
    signal = SilenceSignal()
    
    # Create dummy audio (1 second of silence, 1 second of noise)
    sr = 22050
    t = np.linspace(0, 1, sr)
    noise = np.random.normal(0, 1, sr) * 0.5
    silence = np.zeros(sr)
    
    # Concatenate silence + noise + silence
    audio = np.concatenate([silence, noise, silence])
    
    res = signal.run((audio, sr))
    
    assert 0.0 <= res.score <= 1.0
    assert "silence_ratio" in res.metadata
    
    # Total duration should be approx 3 seconds
    assert abs(res.metadata["total_duration"] - 3.0) < 0.1
    
    print("Silence tests passed!")

if __name__ == "__main__":
    test_silence()
