import sys
from unittest.mock import MagicMock

# Mock heavy dependencies globally before they are imported by package __init__ files
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["cv2"] = MagicMock()
# numpy and scipy are installed in the env, so we don't mock them to avoid structure errors

import unittest
from unittest.mock import patch, mock_open
import os
from veridex.core.provenance import C2PASignal
# Import directly
from veridex.image.provenance import C2PAImageProvenance
from veridex.text.provenance import C2PATextProvenance
from veridex.audio.provenance import C2PAAudioProvenance

class TestC2PASignal(unittest.TestCase):
    def setUp(self):
        self.signal = C2PASignal()

    def test_name_and_dtype(self):
        self.assertEqual(self.signal.name, "c2pa_provenance")
        self.assertEqual(self.signal.dtype, "file")

    def test_subclasses(self):
        self.assertEqual(C2PAImageProvenance().name, "c2pa_image_provenance")
        self.assertEqual(C2PATextProvenance().name, "c2pa_text_provenance")
        self.assertEqual(C2PAAudioProvenance().name, "c2pa_audio_provenance")

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

    def test_run_sidecar_logic(self):
        """Test detection using a sidecar manifest."""
        mock_c2pa = MagicMock()
        mock_reader_instance = MagicMock()
        mock_c2pa.Reader.return_value.__enter__.return_value = mock_reader_instance

        # Mock manifest found via sidecar
        mock_reader_instance.get_active_manifest.return_value = {
            "assertions": [
                {
                    "label": "stds.iptc.digitalSourceType",
                    "data": {"val": "trainedAlgorithmicMedia"}
                }
            ]
        }

        sidecar_content = b"fake_manifest_bytes"

        with patch.dict(sys.modules, {'c2pa': mock_c2pa}):
            with patch("os.path.exists") as mock_exists:
                # Simulate "doc.txt.c2pa" exists
                def side_effect(path):
                    return path == "doc.txt.c2pa"
                mock_exists.side_effect = side_effect

                with patch("builtins.open", mock_open(read_data=sidecar_content)) as mock_file:
                    text_signal = C2PATextProvenance()
                    result = text_signal.run("doc.txt")

                    # Verify score
                    self.assertEqual(result.score, 1.0)
                    self.assertTrue(result.metadata["sidecar_used"])

                    # Verify file was opened
                    mock_file.assert_called_with("doc.txt.c2pa", 'rb')

                    # Verify Reader was called with manifest_data
                    mock_c2pa.Reader.assert_called_with("doc.txt", manifest_data=sidecar_content)

    def test_run_no_manifest(self):
        """Test detection when no active manifest is found."""
        mock_c2pa = MagicMock()
        mock_reader_instance = MagicMock()
        mock_c2pa.Reader.return_value.__enter__.return_value = mock_reader_instance

        mock_reader_instance.get_active_manifest.return_value = None

        with patch.dict(sys.modules, {'c2pa': mock_c2pa}):
            result = self.signal.run("fake_none.jpg")
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.metadata["status"], "no_active_manifest")

if __name__ == "__main__":
    unittest.main()
