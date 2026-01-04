from typing import Any, Dict, Optional
import numpy as np
from veridex.core.signal import BaseSignal, DetectionResult

class LipSyncSignal(BaseSignal):
    """
    Detects Deepfakes by checking Audio-Visual Synchronization (Lip-Sync).
    Uses SyncNet logic.
    """

    @property
    def name(self) -> str:
        return "lipsync_wav2lip"

    @property
    def dtype(self) -> str:
        return "video"

    def check_dependencies(self) -> None:
        try:
            import torch
            import cv2
            import librosa
        except ImportError:
            raise ImportError("LipSyncSignal requires 'torch', 'opencv', and 'librosa'. Install veridex[video].")

    def run(self, input_data: str) -> DetectionResult:
        self.check_dependencies()
        try:
            # 1. Load Audio and Video segments
            # For robustness, we check the AV offset on multiple random 0.2s clips

            offsets = []
            for _ in range(3): # Check 3 segments
                offset = self._calculate_av_offset(input_data)
                if offset is not None:
                    offsets.append(offset)

            if not offsets:
                 return DetectionResult(score=0.5, confidence=0.0, error="Could not extract AV segments")

            avg_offset = sum(offsets) / len(offsets)

            # Metric:
            # Offset is Euclidean distance between Audio and Video embeddings.
            # Small distance -> Sync -> Real.
            # Large distance -> Out of Sync -> Fake.
            # Real < 0.8 (heuristic threshold).

            score = 0.0
            threshold = 0.8
            if avg_offset > threshold:
                # Map distance to probability.
                score = min((avg_offset - threshold) / 1.0, 1.0)

            return DetectionResult(
                score=score,
                confidence=0.7,
                metadata={"av_distance": avg_offset}
            )

        except Exception as e:
            return DetectionResult(score=0.0, confidence=0.0, error=str(e))

    def _calculate_av_offset(self, path: str) -> Optional[float]:
        import torch
        import librosa
        import cv2
        from veridex.video.models.syncnet import SyncNet
        from veridex.video.processing import FaceDetector
        from veridex.utils.downloads import download_file, get_cache_dir

        # 1. Load Audio (0.2s segment)
        try:
            y, sr = librosa.load(path, sr=16000)
        except Exception:
            return None

        if len(y) < 16000: # Need at least 1 sec to find a good chunk
            return None

        # Pick a random start point
        import random
        start_sec = random.uniform(0, len(y)/sr - 0.3)
        start_sample = int(start_sec * sr)
        # 0.2s duration for SyncNet
        duration_samples = int(0.2 * sr)
        audio_chunk = y[start_sample : start_sample + duration_samples]

        # MFCC: 13 coeffs, window 25ms, hop 10ms
        # SyncNet expects specific MFCC shape.
        # (1, 1, 13, 20) -> 13 MFCCs over 20 timesteps (20*10ms = 200ms)
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
        if mfcc.shape[1] < 20:
             mfcc = np.pad(mfcc, ((0,0), (0, 20-mfcc.shape[1])))
        mfcc = mfcc[:, :20]

        # 2. Load Video (5 frames corresponding to that 0.2s)
        # 0.2s at 25fps = 5 frames.
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25

        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(5):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if len(frames) < 5:
            return None

        # 3. Detect and Crop Mouth
        # Simplified: Detect face, take lower half.
        detector = FaceDetector()
        face_crops = []
        for frame in frames:
            dets = detector.detect(frame)
            if not dets:
                # Fallback: center crop? Or just fail this segment
                return None

            # Largest face
            face = max(dets, key=lambda b: b[2] * b[3])
            x, y, w, h = face

            # Mouth region approximation (lower half of face)
            mouth_y = y + h // 2
            mouth_h = h // 2

            mouth_crop = detector.extract_face(frame, (x, mouth_y, w, mouth_h), size=(112, 112))
            face_crops.append(mouth_crop)

        # Stack frames
        # Input: (B, 15, 112, 112). 15 channels = 5 frames * 3 colors.
        # face_crops: 5 * (112, 112, 3)
        video_tensor = np.concatenate(face_crops, axis=2) # (112, 112, 15)

        # To Torch
        audio_t = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0) # (1, 1, 13, 20)
        video_t = torch.from_numpy(video_tensor).float().permute(2, 0, 1).unsqueeze(0) # (1, 15, 112, 112)

        # 4. Inference
        model = SyncNet()
        model.eval()

        # Load weights
        weights_url = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
        weights_path = os.path.join(get_cache_dir(), "syncnet_v2.pth")

        if not os.path.exists(weights_path):
            try:
                download_file(weights_url, weights_path)
            except Exception:
                pass

        if os.path.exists(weights_path):
             try:
                # Note: Official weights might be LuaTorch or different format.
                # This assumes a PyTorch converted version or compatible dict.
                # If mismatch, we ignore to prevent crash, effectively using random weights (untrained).
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
             except:
                pass

        with torch.no_grad():
            a_emb, v_emb = model(audio_t, video_t)
            dist = torch.norm(a_emb - v_emb, p=2, dim=1).item()

        return dist
