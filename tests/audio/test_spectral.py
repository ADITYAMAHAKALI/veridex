"""
Tests for the spectral audio detector.
"""

import pytest
import numpy as np
from pathlib import Path
from veridex.audio.spectral import SpectralSignal


def _has_audio_deps() -> bool:
    """Check if audio dependencies are installed."""
    try:
        import librosa
        import soundfile
        return True
    except ImportError:
        return False


class TestSpectralSignal:
    """Test suite for SpectralSignal."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = SpectralSignal()
        assert detector.name == "spectral_audio_detector"
        assert detector.dtype == "audio"
        assert detector.target_sr == 16000
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        detector = SpectralSignal(
            target_sr=22050,
            n_fft=4096,
            hop_length=1024
        )
        assert detector.target_sr == 22050
        assert detector.n_fft == 4096
        assert detector.hop_length == 1024
    
    def test_invalid_input_type(self):
        """Test handling of invalid input types."""
        detector = SpectralSignal()
        result = detector.run(12345)  # Not a file path
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.error is not None
        assert "file path" in result.error.lower()
    
    def test_missing_dependencies_check(self):
        """Test dependency checking."""
        detector = SpectralSignal()
        # This will either pass or raise ImportError
        # We can't easily mock this without the deps installed
        try:
            detector.check_dependencies()
        except ImportError as e:
            assert "librosa" in str(e) or "soundfile" in str(e)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        detector = SpectralSignal()
        result = detector.run("/nonexistent/path/to/audio.wav")
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.error is not None
    
    @pytest.mark.skipif(
        not _has_audio_deps(),
        reason="Audio dependencies not installed"
    )
    def test_synthetic_audio(self):
        """Test with synthetic audio signal."""
        # Create a temporary audio file
        import tempfile
        import soundfile as sf
        
        # Generate synthetic sine wave (5 seconds)
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            temp_path = f.name
        
        try:
            detector = SpectralSignal()
            result = detector.run(temp_path)
            
            # Should return a result
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert result.error is None
            
            # Check metadata
            assert "spectral_rolloff" in result.metadata
            assert "high_freq_energy" in result.metadata
            assert "high_freq_entropy" in result.metadata
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    @pytest.mark.skipif(
        not _has_audio_deps(),
        reason="Audio dependencies not installed"
    )
    def test_score_computation(self):
        """Test score computation with known features."""
        detector = SpectralSignal()
        
        # Test with AI-like features (low high-freq energy)
        features_ai = {
            "high_freq_energy": 3.0,
            "spectral_rolloff": 3500,
            "high_freq_entropy": 2.5,
            "high_freq_stability": 0.8
        }
        score_ai = detector._compute_score(features_ai)
        assert score_ai > 0.5  # Should indicate AI
        
        # Test with human-like features (high high-freq energy)
        features_human = {
            "high_freq_energy": 15.0,
            "spectral_rolloff": 7000,
            "high_freq_entropy": 6.0,
            "high_freq_stability": 3.0
        }
        score_human = detector._compute_score(features_human)
        assert score_human < 0.3  # Should indicate human
