"""
Tests for core provenance module (C2PASignal).
"""

import pytest
from unittest.mock import patch, MagicMock
from veridex.core.provenance import C2PASignal
from veridex.core.signal import DetectionResult


class TestC2PASignal:
    """Test suite for C2PASignal."""

    def test_initialization(self):
        """Test that C2PASignal initializes correctly."""
        signal = C2PASignal()
        assert signal is not None

    def test_name_property(self):
        """Test that name property returns correct value."""
        signal = C2PASignal()
        assert signal.name == "c2pa_provenance"

    def test_dtype_property(self):
        """Test that dtype property returns correct value."""
        signal = C2PASignal()
        assert signal.dtype == "file"

    def test_check_dependencies_missing(self):
        """Test dependency check when c2pa is not installed."""
        signal = C2PASignal()
        
        with patch.dict('sys.modules', {'c2pa': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError) as exc_info:
                    signal.check_dependencies()
                
                assert "c2pa" in str(exc_info.value).lower()
                assert "pip install c2pa-python" in str(exc_info.value)

    def test_check_dependencies_success(self):
        """Test dependency check when c2pa is installed."""
        signal = C2PASignal()
        
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            # Should not raise
            signal.check_dependencies()

    def test_run_invalid_input_type(self):
        """Test run with non-string input."""
        signal = C2PASignal()
        
        # Test with integer
        result = signal.run(123)
        assert isinstance(result, DetectionResult)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert "Input must be a file path string" in result.error

        # Test with list
        result = signal.run([1, 2, 3])
        assert result.score == 0.0
        assert result.error is not None

    def test_run_missing_dependencies(self):
        """Test run when c2pa is not installed."""
        signal = C2PASignal()
        
        with patch.dict('sys.modules', {'c2pa': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                result = signal.run("test.jpg")
                
                assert isinstance(result, DetectionResult)
                assert result.score == 0.0
                assert result.confidence == 0.0
                assert "c2pa-python not installed" in result.error

    def test_run_no_manifest(self):
        """Test run when file has no C2PA manifest."""
        signal = C2PASignal()
        
        mock_c2pa = MagicMock()
        mock_c2pa.read_json.return_value = None
        
        with patch.dict('sys.modules', {'c2pa': mock_c2pa}):
            result = signal.run("test.jpg")
            
            assert isinstance(result, DetectionResult)
            assert result.score == 0.0
            assert result.confidence == 1.0
            assert result.metadata["status"] == "no_manifest"
            assert result.error is None

    def test_run_with_string_path(self):
        """Test that run accepts string path input."""
        signal = C2PASignal()
        
        # Should accept string path (will likely fail to find c2pa, but that's ok)
        result = signal.run("/path/to/test.jpg")
        
        assert isinstance(result, DetectionResult)
        # Either error due to missing library or file not found
        assert result.error is not None or result.score >= 0








