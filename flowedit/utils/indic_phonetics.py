"""
Phonetic Normalizer & Respeller Utility for Indic, Devanagari, and Medical Terms.

Resolves tokenizer BPE artifacts for:
1. Devanagari & Indic text (मृण्मयी, साखरवाडे, Mrunmayee, Sakharwade).
2. Medical / Technical terms (e.g. Anophthalmia, Otorhinolaryngology, Pembrolizumab).
3. English BPE abbreviation artifacts (e.g. "Mr" at start of Indic names -> "Mroo").
"""

import re

# Devanagari Unicode Range
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]+')

# Devanagari to Latin phonetic transliteration dictionary
DEVANAGARI_MAP = {
    "मृण्मयी": "Mrunmayee",
    "साखरवाडे": "Sakharwade",
    "कृष्णमूर्ति": "Krishnamurthy",
    "तिरुवनंतपुरम": "Thiruvananthapuram",
    "अरविन्द": "Aurobindo",
    "भुवनेश्वर": "Bhubaneswar",
}

# Medical & Technical Phonetic Normalizations
MEDICAL_TERMS_MAP = [
    (re.compile(r'\bAnophthalmia\b', re.IGNORECASE), "An-oph-thal-mi-a"),
    (re.compile(r'\bOtorhinolaryngology\b', re.IGNORECASE), "O-to-rhi-no-la-ryn-gol-o-gy"),
    (re.compile(r'\bPembrolizumab\b', re.IGNORECASE), "Pem-bro-liz-u-mab"),
    (re.compile(r'\bLevetiracetam\b', re.IGNORECASE), "Le-ve-ti-ra-ce-tam"),
    (re.compile(r'\bCholedochoduodenostomy\b', re.IGNORECASE), "Cho-le-do-cho-du-o-de-nos-to-my"),
]

# Indic Name BPE Respelling Rules (prevents "Mr" -> "Mister / Emran")
INDIC_PREFIX_RESPEL = [
    (re.compile(r'\bMr(un|oo|u)', re.IGNORECASE), r'Mroo\1'),
    (re.compile(r'\bMr(in|i)', re.IGNORECASE), r'Mree\1'),
    (re.compile(r'\bMr(it|y)', re.IGNORECASE), r'Mree\1'),
    (re.compile(r'\bSakh([a-z]+)wade', re.IGNORECASE), r'Sakh-\1-waade'),
    (re.compile(r'\bThiruvananthapuram\b', re.IGNORECASE), r'Thi-ru-va-nan-tha-pu-ram'),
    (re.compile(r'\bVisakhapatnam\b', re.IGNORECASE), r'Vi-sa-kha-pat-nam'),
]

def normalize_indic_phonetics(text: str) -> str:
    """Normalize Indic, Devanagari, and Medical terms for clean TTS tokenization.
    
    Examples:
        >>> normalize_indic_phonetics("मृण्मयी साखरवाडे")
        "Mroonmayee Sakhar-waade"
        >>> normalize_indic_phonetics("My name is Mrunmayee Sakharwade")
        "My name is Mroonmayee Sakh-ar-waade"
    """
    normalized = text.strip()

    # 1. Map direct Devanagari script words to phonetic Latin equivalents
    for dev_word, lat_word in DEVANAGARI_MAP.items():
        if dev_word in normalized:
            normalized = normalized.replace(dev_word, lat_word)

    # 2. Apply Medical terms syllabification
    for pattern, replacement in MEDICAL_TERMS_MAP:
        normalized = pattern.sub(replacement, normalized)

    # 3. Apply Indic name prefix BPE respelling rules
    for pattern, replacement in INDIC_PREFIX_RESPEL:
        normalized = pattern.sub(replacement, normalized)

    # 4. Basic digit normalization to prevent TTS tokenization failures (silence / hallucination)
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    def replace_digits(match):
        return " ".join([digit_map[d] for d in match.group(0)])
        
    normalized = re.sub(r'\d+', replace_digits, normalized)

    return normalized
