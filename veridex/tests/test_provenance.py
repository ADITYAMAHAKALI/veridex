import unittest
from unittest.mock import MagicMock, patch
import sys
from veridex.core.provenance import C2PASignal

class TestC2PASignal(unittest.TestCase):
    def setUp(self):
        self.signal = C2PASignal()

    def test_name_and_dtype(self):
        self.assertEqual(self.signal.name, "c2pa_provenance")
        self.assertEqual(self.signal.dtype, "file")

    def test_run_ai_manifest(self):
        """Test detection when manifest indicates AI generation."""
        mock_c2pa = MagicMock()
        mock_reader_instance = MagicMock()
        mock_c2pa.Reader.return_value.__enter__.return_value = mock_reader_instance

        # Mock active manifest with AI source type
        mock_reader_instance.get_active_manifest.return_value = {
            "assertions": [
                {
                    "label": "stds.iptc.digitalSourceType",
                    "data": {
                        "val": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                    }
                }
            ]
        }

        with patch.dict(sys.modules, {'c2pa': mock_c2pa}):
            result = self.signal.run("fake_ai.jpg")
            self.assertEqual(result.score, 1.0)
            self.assertEqual(result.confidence, 1.0)
            self.assertTrue(result.metadata["is_ai_signed"])

    def test_run_human_manifest(self):
        """Test detection when manifest indicates Human/Camera source."""
        mock_c2pa = MagicMock()
        mock_reader_instance = MagicMock()
        mock_c2pa.Reader.return_value.__enter__.return_value = mock_reader_instance

        # Mock active manifest with normal source type (or missing)
        mock_reader_instance.get_active_manifest.return_value = {
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {"actions": [{"action": "c2pa.action.created"}]}
                }
            ]
        }

        with patch.dict(sys.modules, {'c2pa': mock_c2pa}):
            result = self.signal.run("fake_human.jpg")
            self.assertEqual(result.score, 0.0)
            self.assertFalse(result.metadata["is_ai_signed"])

    def test_run_no_manifest(self):
        """Test detection when no active manifest is found."""
        mock_c2pa = MagicMock()
        mock_reader_instance = MagicMock()
        mock_c2pa.Reader.return_value.__enter__.return_value = mock_reader_instance

        # Return None for manifest
        mock_reader_instance.get_active_manifest.return_value = None

        with patch.dict(sys.modules, {'c2pa': mock_c2pa}):
            result = self.signal.run("fake_none.jpg")
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.metadata["status"], "no_active_manifest")

    def test_run_read_error(self):
        """Test handling of read errors (e.g. invalid file format)."""
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.side_effect = Exception("File format not supported")

        with patch.dict(sys.modules, {'c2pa': mock_c2pa}):
            result = self.signal.run("invalid.txt")
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.metadata["status"], "read_error")
            self.assertIn("File format not supported", result.metadata["details"])

    def test_run_invalid_input(self):
        """Test with non-string input."""
        result = self.signal.run(123)
        self.assertEqual(result.error, "Input must be a file path string.")

if __name__ == "__main__":
    unittest.main()
