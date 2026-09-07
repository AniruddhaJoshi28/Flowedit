try:
    from .hopfield_memory import HopfieldMemory
    ModernHopfieldMemory = HopfieldMemory
except ImportError:
    HopfieldMemory = None
    ModernHopfieldMemory = None

from .s3_storage import S3SpellingStore, s3_spelling_store

