try:
    from .whisper_aligner import WhisperAligner, AlignmentResult
except ImportError:
    WhisperAligner = None
    AlignmentResult = None

from .homograph_resolver import HomographContextResolver, SenseProfile
