"""
Tests for audio utility functions.
"""

import pytest
import numpy as np
from pathlib import Path


def _has_audio_deps() -> bool:
    """Check if audio dependencies are installed."""
    try:
        import librosa
        import soundfile
        return True
    except ImportError:
        return False


class TestAudioUtils:
    """Test suite for audio utilities."""
    
    @pytest.mark.skipif(
        not _has_audio_deps(),
        reason="Audio dependencies not installed"
    )
    def test_load_audio(self):
        """Test audio loading."""
        from veridex.audio.utils import load_audio
        import tempfile
        import soundfile as sf
        
        # Create test audio
        sr = 16000
        duration = 2.0
        audio_data = np.random.randn(int(sr * duration)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sr)
            temp_path = f.name
        
        try:
            audio, loaded_sr = load_audio(temp_path, target_sr=16000)
            
            assert loaded_sr == 16000
            assert len(audio) > 0
            assert audio.dtype == np.float32
            assert np.max(np.abs(audio)) <= 1.0  # Normalized
            
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.skipif(
        not _has_audio_deps(),
        reason="Audio dependencies not installed"
    )
    def test_extract_mel_spectrogram(self):
        """Test mel-spectrogram extraction."""
        from veridex.audio.utils import extract_mel_spectrogram
        
        # Create synthetic audio
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        mel_spec = extract_mel_spectrogram(audio, sr=sr, n_mels=128)
        
        assert mel_spec.shape[0] == 128  # n_mels
        assert mel_spec.shape[1] > 0  # time frames
    
    @pytest.mark.skipif(
        not _has_audio_deps(),
        reason="Audio dependencies not installed"
    )
    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        from veridex.audio.utils import extract_mfcc
        
        # Create synthetic audio
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        mfcc = extract_mfcc(audio, sr=sr, n_mfcc=40)
        
        assert mfcc.shape[0] == 40  # n_mfcc
        assert mfcc.shape[1] > 0  # time frames
    
    @pytest.mark.skipif(
        not _has_audio_deps(),
        reason="Audio dependencies not installed"
    )
    def test_validate_audio(self):
        """Test audio validation."""
        from veridex.audio.utils import validate_audio
        
        sr = 16000
        
        # Valid audio
        valid_audio = np.random.randn(sr * 2).astype(np.float32)  # 2 seconds
        is_valid, error = validate_audio(valid_audio, sr)
        assert is_valid
        assert error is None
        
        # Too short
        short_audio = np.random.randn(sr // 4).astype(np.float32)  # 0.25 seconds
        is_valid, error = validate_audio(short_audio, sr, min_duration=0.5)
        assert not is_valid
        assert "too short" in error.lower()
        
        # Silent
        silent_audio = np.zeros(sr * 2).astype(np.float32)
        is_valid, error = validate_audio(silent_audio, sr)
        assert not is_valid
        assert "silent" in error.lower()
        
        # Invalid values
        invalid_audio = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
        is_valid, error = validate_audio(invalid_audio, sr, min_duration=0.0)
        assert not is_valid
        assert "invalid" in error.lower()

