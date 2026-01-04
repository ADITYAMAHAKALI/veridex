from typing import List, Tuple, Optional
import numpy as np

class FaceDetector:
    """
    A lightweight face detector using OpenCV's Haar Cascades.
    This avoids heavy dependencies like dlib or face_recognition for the base case.
    """
    def __init__(self):
        try:
            import cv2
        except ImportError:
            raise ImportError("FaceDetector requires 'opencv-python-headless'. Please install veridex[video].")

        self.cv2 = cv2
        # Use the default haarcascade for frontal face
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.

        Args:
            frame: RGB or BGR numpy array (OpenCV uses BGR).

        Returns:
            List of (x, y, w, h) tuples.
        """
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=self.cv2.CASCADE_SCALE_IMAGE
        )
        # Convert to list of tuples
        return [tuple(f) for f in faces]

    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Extract and resize the face ROI.
        """
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=frame.dtype)
        return self.cv2.resize(face, size)
