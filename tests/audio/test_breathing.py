"""
Tests for breathing-based audio deepfake detector.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from veridex.audio.breathing_signal import BreathingSignal
from veridex.core.signal import DetectionResult


class TestBreathingSignal:
    """Test suite for BreathingSignal."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        signal = BreathingSignal()
        assert signal.target_sr == 16000

    def test_initialization_custom(self):
        """Test initialization with custom target sample rate."""
        signal = BreathingSignal(target_sr=22050)
        assert signal.target_sr == 22050

    def test_name_property(self):
        """Test name property."""
        signal = BreathingSignal()
        assert signal.name == "breathing_audio_detector"

    def test_dtype_property(self):
        """Test dtype property."""
        signal = BreathingSignal()
        assert signal.dtype == "audio"

    def test_check_dependencies_missing_librosa(self):
        """Test dependency check when librosa is missing."""
        signal = BreathingSignal()
        
        with patch.dict('sys.modules', {'librosa': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError) as exc_info:
                    signal.check_dependencies()
                
                assert "librosa" in str(exc_info.value).lower()

    def test_check_dependencies_success(self):
        """Test dependency check when all dependencies are present."""
        signal = BreathingSignal()
        
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            # Should not raise
            signal.check_dependencies()

    def test_run_invalid_input_type(self):
        """Test run with invalid input types."""
        signal = BreathingSignal()
        
        # Test with integer
        result = signal.run(123)
        assert isinstance(result, DetectionResult)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert "Input must be a file path" in result.error

        # Test with list
        result = signal.run([1, 2, 3])
        assert result.error is not None

    def test_run_missing_dependencies(self):
        """Test run when dependencies are missing."""
        signal = BreathingSignal()
        
        with patch.object(signal, 'check_dependencies', side_effect=ImportError("librosa not found")):
            result = signal.run("test.wav")
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.0
            assert result.confidence == 0.0
            assert "librosa not found" in result.error







    def test_compute_breath_metrics(self):
        """Test breath metrics computation."""
        signal = BreathingSignal()
        
        # Sample breaths: (start_time, end_time)
        breaths = [(1.0, 1.3), (3.0, 3.4), (5.5, 5.8)]
        duration = 10.0
        
        metrics = signal._compute_breath_metrics(breaths, duration)
        
        assert "num_breaths" in metrics
        assert metrics["num_breaths"] == 3
        assert "breaths_per_minute" in metrics
        assert metrics["breaths_per_minute"] == 18.0  # 3 breaths in 10s = 18 BPM
        assert "breath_ratio" in metrics
        assert "avg_breath_duration" in metrics
        assert "interval_std" in metrics
        assert "duration" in metrics

    def test_compute_breath_metrics_no_breaths(self):
        """Test metrics computation with no breaths."""
        signal = BreathingSignal()
        
        breaths = []
        duration = 10.0
        
        metrics = signal._compute_breath_metrics(breaths, duration)
        
        assert metrics["num_breaths"] == 0
        assert metrics["breaths_per_minute"] == 0
        assert metrics["avg_breath_duration"] == 0

    def test_compute_score_very_low_bpm(self):
        """Test score computation with very low BPM (strong AI indicator)."""
        signal = BreathingSignal()
        
        metrics = {
            "duration": 10.0,
            "breaths_per_minute": 0.5,
            "num_breaths": 1
        }
        
        score = signal._compute_score(metrics)
        
        # Very low BPM should give high AI score
        assert score >= 0.8

    def test_compute_score_normal_bpm(self):
        """Test score computation with normal BPM (human indicator)."""
        signal = BreathingSignal()
        
        metrics = {
            "duration": 10.0,
            "breaths_per_minute": 15.0,
            "num_breaths": 3
        }
        
        score = signal._compute_score(metrics)
        
        # Normal BPM should give low AI score
        assert score <= 0.3

    def test_compute_score_short_audio(self):
        """Test score computation with short audio (unreliable)."""
        signal = BreathingSignal()
        
        metrics = {
            "duration": 2.0,
            "breaths_per_minute": 0.0,
            "num_breaths": 0
        }
        
        score = signal._compute_score(metrics)
        
        # Short audio should return neutral score
        assert score == 0.5

    def test_compute_confidence_short_audio(self):
        """Test confidence estimation with short audio."""
        signal = BreathingSignal()
        
        confidence = signal._compute_confidence({}, 3.0)
        
        # Short audio should have low confidence
        assert confidence < 0.5

    def test_compute_confidence_long_audio(self):
        """Test confidence estimation with long audio."""
        signal = BreathingSignal()
        
        confidence = signal._compute_confidence({}, 25.0)
        
        # Long audio should have high confidence
        assert confidence >= 0.8


