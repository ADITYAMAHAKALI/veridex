"""
Tests for AASIST-inspired audio deepfake detector.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from veridex.audio.aasist_signal import AASISTSignal
from veridex.core.signal import DetectionResult


class TestAASISTSignal:
    """Test suite for AASISTSignal."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        signal = AASISTSignal()
        assert signal.target_sr == 16000
        assert signal.n_fft == 512
        assert signal.hop_length == 256

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        signal = AASISTSignal(target_sr=22050, n_fft=1024, hop_length=512)
        assert signal.target_sr == 22050
        assert signal.n_fft == 1024
        assert signal.hop_length == 512

    def test_name_property(self):
        """Test name property."""
        signal = AASISTSignal()
        assert signal.name == "aasist_audio_detector"

    def test_dtype_property(self):
        """Test dtype property."""
        signal = AASISTSignal()
        assert signal.dtype == "audio"

    def test_check_dependencies_missing(self):
        """Test dependency check when dependencies are missing."""
        signal = AASISTSignal()
        
        with patch.dict('sys.modules', {'librosa': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError) as exc_info:
                    signal.check_dependencies()
                
                assert "librosa" in str(exc_info.value).lower()

    def test_check_dependencies_success(self):
        """Test dependency check when all dependencies are present."""
        signal = AASISTSignal()
        
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            # Should not raise
            signal.check_dependencies()

    def test_run_invalid_input_type(self):
        """Test run with invalid input types."""
        signal = AASISTSignal()
        
        # Test with integer
        result = signal.run(123)
        assert isinstance(result, DetectionResult)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert "Input must be a file path" in result.error

        # Test with dict
        result = signal.run({"audio": "data"})
        assert result.error is not None

    def test_run_missing_dependencies(self):
        """Test run when dependencies are missing."""
        signal = AASISTSignal()
        
        with patch.object(signal, 'check_dependencies', side_effect=ImportError("librosa not found")):
            result = signal.run("test.wav")
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.0
            assert result.confidence == 0.0
            assert "librosa not found" in result.error





    def test_extract_spectro_temporal_features(self):
        """Test spectro-temporal feature extraction."""
        signal = AASISTSignal()
        
        # Create synthetic mel spectrogram and audio
        mel_spec = np.random.randn(80, 100)
        audio = np.random.randn(32000)
        sr = 16000
        
        with patch('scipy.stats.entropy', return_value=5.0):
            with patch('scipy.signal.stft') as mock_stft:
                # Mock STFT output
                f = np.arange(257)
                t = np.arange(100)
                Zxx = np.random.randn(257, 100) + 1j * np.random.randn(257, 100)
                mock_stft.return_value = (f, t, Zxx)
                
                features = signal._extract_spectro_temporal_features(mel_spec, audio, sr)
        
        # Verify all expected features are present
        assert "mean_temporal_variation" in features
        assert "max_temporal_variation" in features
        assert "mean_spectral_variation" in features
        assert "phase_coherence" in features
        assert "phase_std" in features
        assert "energy_entropy" in features
        assert "energy_uniformity" in features
        assert "mean_band_correlation" in features
        assert "mean_spectral_flux" in features
        
        # Verify feature values are numeric
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)

    def test_extract_spectro_temporal_features_uniform_audio(self):
        """Test feature extraction with uniform audio (AI indicator)."""
        signal = AASISTSignal()
        
        # Create uniform mel spectrogram (AI-like)
        mel_spec = np.ones((80, 100)) * 0.5
        audio = np.ones(32000) * 0.1
        sr = 16000
        
        with patch('scipy.stats.entropy', return_value=0.1):  # Low entropy
            with patch('scipy.signal.stft') as mock_stft:
                f = np.arange(257)
                t = np.arange(100)
                Zxx = np.ones((257, 100), dtype=complex)
                mock_stft.return_value = (f, t, Zxx)
                
                features = signal._extract_spectro_temporal_features(mel_spec, audio, sr)
        
        # Uniform audio should have low variation
        assert features["mean_temporal_variation"] < 1.0
        assert features["mean_spectral_variation"] < 1.0

    def test_compute_score_ai_indicators(self):
        """Test score computation with strong AI indicators."""
        signal = AASISTSignal()
        
        features = {
            "mean_temporal_variation": 5.0,  # Low (AI indicator)
            "max_temporal_variation": 10.0,
            "mean_spectral_variation": 3.0,
            "phase_coherence": 0.5,
            "phase_std": 0.1,
            "energy_entropy": 3.0,
            "energy_uniformity": 0.7,  # High (AI indicator)
            "mean_band_correlation": 0.85,  # High (AI indicator)
            "mean_spectral_flux": 10.0,  # Low (AI indicator)
        }
        
        score = signal._compute_score(features)
        
        # Strong AI indicators should give high score
        assert score >= 0.5

    def test_compute_score_human_indicators(self):
        """Test score computation with human speech indicators."""
        signal = AASISTSignal()
        
        features = {
            "mean_temporal_variation": 18.0,  # High (human indicator)
            "max_temporal_variation": 30.0,
            "mean_spectral_variation": 15.0,
            "phase_coherence": 2.0,  # Normal range
            "phase_std": 0.5,
            "energy_entropy": 5.0,
            "energy_uniformity": 0.25,  # Low (human indicator)
            "mean_band_correlation": 0.5,  # Normal range
            "mean_spectral_flux": 25.0,  # High (human indicator)
        }
        
        score = signal._compute_score(features)
        
        # Human indicators should give low score
        assert score <= 0.3

    def test_compute_score_boundary_values(self):
        """Test score computation with boundary values."""
        signal = AASISTSignal()
        
        # Features at boundaries
        features = {
            "mean_temporal_variation": 8.0,  # At threshold
            "max_temporal_variation": 15.0,
            "mean_spectral_variation": 10.0,
            "phase_coherence": 1.5,
            "phase_std": 0.3,
            "energy_entropy": 4.0,
            "energy_uniformity": 0.4,  # At threshold
            "mean_band_correlation": 0.5,
            "mean_spectral_flux": 15.0,  # At threshold
        }
        
        score = signal._compute_score(features)
        
        # Should return valid score
        assert 0.0 <= score <= 1.0

    def test_estimate_confidence_short_audio(self):
        """Test confidence estimation with short audio."""
        signal = AASISTSignal()
        
        audio = np.random.randn(8000)  # 0.5 seconds
        features = {"energy_uniformity": 0.5}
        sr = 16000
        
        confidence = signal._estimate_confidence(audio, features, sr)
        
        # Short audio should have reduced confidence
        assert confidence < 0.65

    def test_estimate_confidence_long_audio(self):
        """Test confidence estimation with long audio."""
        signal = AASISTSignal()
        
        audio = np.random.randn(96000)  # 6 seconds
        features = {"energy_uniformity": 0.5}
        sr = 16000
        
        confidence = signal._estimate_confidence(audio, features, sr)
        
        # Long audio should have higher confidence
        assert confidence > 0.65

    def test_estimate_confidence_decisive_features(self):
        """Test confidence boost with decisive features."""
        signal = AASISTSignal()
        
        audio = np.random.randn(96000)  # 6 seconds
        features = {"energy_uniformity": 0.7}  # Very uniform (decisive)
        sr = 16000
        
        confidence = signal._estimate_confidence(audio, features, sr)
        
        # Decisive features should boost confidence
        assert confidence >= 0.7

    def test_estimate_confidence_bounds(self):
        """Test that confidence stays within [0, 1] bounds."""
        signal = AASISTSignal()
        
        # Extreme case
        audio = np.random.randn(160000)  # 10 seconds
        features = {"energy_uniformity": 0.9}
        sr = 16000
        
        confidence = signal._estimate_confidence(audio, features, sr)
        
        # Should be capped at 1.0
        assert 0.0 <= confidence <= 1.0



    def test_feature_extraction_with_nan_correlation(self):
        """Test feature extraction handles NaN in correlation calculation."""
        signal = AASISTSignal()
        
        # Create mel spec with constant bands that would produce NaN correlations
        mel_spec = np.zeros((80, 100))
        mel_spec[0, :] = 1.0  # Constant band
        audio = np.random.randn(32000)
        sr = 16000
        
        with patch('scipy.stats.entropy', return_value=0.5):
            with patch('scipy.signal.stft') as mock_stft:
                f = np.arange(257)
                t = np.arange(100)
                Zxx = np.random.randn(257, 100) + 1j * np.random.randn(257, 100)
                mock_stft.return_value = (f, t, Zxx)
                
                features = signal._extract_spectro_temporal_features(mel_spec, audio, sr)
        
        # Should handle NaN gracefully
        assert not np.isnan(features["mean_band_correlation"])
