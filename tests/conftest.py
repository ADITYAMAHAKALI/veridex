"""
Global pytest configuration and fixtures.

This module sets up mocks for heavy dependencies at the session level
to prevent import issues across all test modules.
"""
import sys
from unittest.mock import MagicMock
import pytest


# Set up global mocks before any imports  
# These need to be set up early to avoid issues with transformers and other packages
#that check for optional dependencies during import

# Mock av with __spec__ to avoid transformers import issues
if "av" not in sys.modules or not hasattr(sys.modules.get("av"), "__spec__"):
    mock_av = MagicMock()
    mock_av.__spec__ = MagicMock()
    sys.modules["av"] = mock_av

# Mock cv2 with __spec__ to avoid transformers import issues, but only if not installed
try:
    import cv2
except ImportError:
    # Only mock if cv2 is not installed
    mock_cv2 = MagicMock()
    mock_cv2.__spec__ = MagicMock()
    sys.modules["cv2"] = mock_cv2

# Mock torch and torchvision with __spec__ to avoid transformers import issues
try:
    import torch
except ImportError:
    mock_torch = MagicMock()
    mock_torch.__spec__ = MagicMock()
    # Also mock submodules that video tests might need
    mock_torch.nn = MagicMock()
    mock_torch.nn.functional = MagicMock()
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = mock_torch.nn
    sys.modules["torch.nn.functional"] = mock_torch.nn.functional

try:
    import torchvision
except ImportError:
    mock_torchvision = MagicMock()
    mock_torchvision.__spec__ = MagicMock()
    sys.modules["torchvision"] = mock_torchvision

# Try to import scipy - if not available, create a minimal mock with __version__
# This allows sklearn imports to work while still allowing tests to patch specific functions
try:
    import scipy
except ImportError:
    # Only mock if scipy is not installed
    mock_scipy = MagicMock()
    mock_scipy.__version__ = "1.11.0"
    sys.modules["scipy"] = mock_scipy
