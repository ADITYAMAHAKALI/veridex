import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from veridex.video.utils import chunk_video_frames, smart_sample_frames, validate_video_file

class TestVideoUtils(unittest.TestCase):
    
    def test_chunk_video_frames(self):
        # Create 10 frames
        frames = np.zeros((10, 100, 100, 3))
        
        # Chunk size 5, no overlap
        chunks = list(chunk_video_frames(frames, chunk_size=5, overlap=0))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1].shape[0], 5)
        self.assertEqual(chunks[1][1].shape[0], 5)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[1][0], 5)
        
        # Chunk size 6, overlap 2
        # Start 0: [0:6] (6 frames)
        # Next start: 0 + (6-2) = 4. [4:10] (6 frames). 
        # Next start: 4 + 4 = 8. [8:14] ... implementation breaks if previous chunk end >= total
        # So it yields [0:6] and [4:10]. Then sees 10 >= 10 and stops.
        chunks = list(chunk_video_frames(frames, chunk_size=6, overlap=2))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1].shape[0], 6)
        self.assertEqual(chunks[1][1].shape[0], 6)
        
    def test_smart_sample_frames(self):
        total = 100
        target = 10
        
        # Uniform
        indices = smart_sample_frames(total, target, 'uniform')
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        
        # Random
        indices = smart_sample_frames(total, target, 'random')
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(set(indices)), 10) # Unique
        
        # Target > Total
        indices = smart_sample_frames(10, 20)
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices, list(range(10)))
        
        # Invalid strategy
        with self.assertRaises(ValueError):
            smart_sample_frames(100, 10, 'invalid')
            
    @patch('cv2.VideoCapture')
    @patch('os.path.exists')
    def test_validate_video_file(self, mock_exists, mock_cap_cls):
        # File not found
        mock_exists.return_value = False
        valid, err, meta = validate_video_file("missing.mp4")
        self.assertFalse(valid)
        self.assertIn("File not found", err)
        
        # File found, setup mock cap
        mock_exists.return_value = True
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        
        # Unable to open
        mock_cap.isOpened.return_value = False
        valid, err, meta = validate_video_file("bad.mp4")
        self.assertFalse(valid)
        self.assertIn("Unable to open", err)
        
        # Valid video
        mock_cap.isOpened.return_value = True
        
        # Mock properties
        # CV2 props are accessed via get(int). Let's mock side effect or return values
        import cv2
        def get_prop(prop_id):
            if prop_id == cv2.CAP_PROP_FPS: return 30.0
            if prop_id == cv2.CAP_PROP_FRAME_COUNT: return 100
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH: return 1920
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return 1080
            return 0
        
        mock_cap.get.side_effect = get_prop
        
        valid, err, meta = validate_video_file("good.mp4")
        self.assertTrue(valid)
        self.assertEqual(meta['fps'], 30.0)
        self.assertEqual(meta['total_frames'], 100)
        
        # Short video
        def get_prop_short(prop_id):
            if prop_id == cv2.CAP_PROP_FPS: return 30.0
            if prop_id == cv2.CAP_PROP_FRAME_COUNT: return 10 # < 30 frames
            return 0
            
        mock_cap.get.side_effect = get_prop_short
        valid, err, meta = validate_video_file("short.mp4")
        self.assertFalse(valid)
        self.assertIn("too short", err)

if __name__ == '__main__':
    unittest.main()
