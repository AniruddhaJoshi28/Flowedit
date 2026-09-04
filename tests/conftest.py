"""
Pytest configuration and environment mocks for test suite.
"""

import sys
from unittest.mock import MagicMock

# Guard against C++ DLL mismatch on Windows for torchaudio
try:
    import torchaudio
except (ImportError, OSError, Exception):
    ta_mock = MagicMock()
    sys.modules["torchaudio"] = ta_mock
    sys.modules["torchaudio.transforms"] = MagicMock()

try:
    import librosa
except (ImportError, OSError, Exception):
    lr_mock = MagicMock()
    lr_mock.get_duration = MagicMock(return_value=1.0)
    sys.modules["librosa"] = lr_mock

try:
    import scipy
except (ImportError, OSError, Exception):
    sys.modules["scipy"] = MagicMock()
    sys.modules["scipy.signal"] = MagicMock()
