"""
Pytest configuration and environment mocks for test suite.
"""

import os
import sys
from pathlib import Path
import uuid
from unittest.mock import MagicMock

import pytest


# Pytest's Windows tmp_path factory creates mode-0700 directories. In this
# managed OneDrive workspace that ACL becomes unreadable to the sandbox token.
# Use an ordinary project-local root with unique per-test directories instead.
_PROJECT_TEST_ROOT = Path(__file__).resolve().parents[1] / "research_test_tmp"
_PROJECT_TEST_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def tmp_path(request):
    safe_name = "".join(c if c.isalnum() else "_" for c in request.node.name)[:48]
    path = _PROJECT_TEST_ROOT / f"{safe_name}_{uuid.uuid4().hex}"
    path.mkdir(parents=False, exist_ok=False)
    return path

# Corrections learned by the sense lexicon during tests go to a throwaway file, never the repo
_lexicon_dir = _PROJECT_TEST_ROOT / f"sense_lexicon_{uuid.uuid4().hex}"
_lexicon_dir.mkdir(parents=False, exist_ok=False)
os.environ["FLOWEDIT_SENSE_LEXICON_USER"] = str(_lexicon_dir / "sense_lexicon_user.pt")

# Let transformers record its optional-dependency availability before any module below is
# replaced by a mock (the contextual sense encoder loads a transformers text model).
try:
    import transformers.utils.import_utils  # noqa: F401
except Exception:
    pass

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
