import torch
import torch.nn as nn

class SyncNet(nn.Module):
    """
    SyncNet: Audio-Visual Synchronization Network.
    Simplified implementation.
    """
    def __init__(self):
        super(SyncNet, self).__init__()

        # Audio Encoder (takes MFCCs)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), # Reduces time dim
        )
        self.audio_fc = nn.Linear(512 * 13 * 1, 1024) # Approximate shape after pool

        # Video Encoder (takes 5 frames of mouth crop, stacked channel-wise)
        # Input: (B, 15, H/2, W/2) ? Usually 5 frames * 3 channels = 15
        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.face_fc = nn.Linear(512 * 7 * 7, 1024) # Assuming 112x112 input -> 7x7 at end

    def forward(self, audio, video):
        # Audio: (B, 1, 13, 20) -> MFCC (13 coeffs, 20 timesteps ~ 0.2s)
        # Video: (B, 15, H, W) -> 5 frames lower half face (112x112)

        a = self.audio_encoder(audio)
        v = self.face_encoder(video)

        # Flatten
        a = a.view(a.size(0), -1)
        v = v.view(v.size(0), -1)

        # Project to shared embedding space
        if a.shape[1] != self.audio_fc.in_features:
            # Handle shape mismatch if input size varies, usually fix input size
            # Just squelch for now or dynamic Linear
            pass

        a = self.audio_fc(a)
        v = self.face_fc(v)

        # L2 Normalize
        a = nn.functional.normalize(a, p=2, dim=1)
        v = nn.functional.normalize(v, p=2, dim=1)

        return a, v
