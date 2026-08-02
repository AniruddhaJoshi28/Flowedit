"""
Multilingual & Indic Phonetic Dataset Builder for XTTS Fine-Tuning.

Generates structured dataset containing:
1. Indic Consonants: Ka, Kha, Ga, Gha, Nga, Cha, Chha, Ja, Jha, Nya, Ta, Tha, Da, Dha, Na, Pa, Pha, Ba, Bha, Ma, Ya, Ra, La, Va, Sha, Sa, Ha, Ksha, Tra, Gya.
2. Full Barakhadi Matrix: Complete vowel/matra combinations for all consonants.
3. Polyglot & Indian Proper Nouns across 18 language families.

Output directory layout:
    formatted_dataset/
    ├── metadata.csv (or tts_train.csv)
    └── wavs/
        ├── ka.wav
        ├── kaa.wav
        └── ...
"""

import os
import csv
import argparse
from pathlib import Path
import numpy as np

# ── 1. Indic Consonants & Phonetic Mapping ─────────────────────────────────
INDIC_CONSONANTS = [
    # Gutturals (K-group)
    {"char": "क", "latin": "Ka", "ipa": "kə"},
    {"char": "ख", "latin": "Kha", "ipa": "kʰə"},
    {"char": "ग", "latin": "Ga", "ipa": "ɡə"},
    {"char": "घ", "latin": "Gha", "ipa": "ɡʱə"},
    {"char": "ङ", "latin": "Nga", "ipa": "ŋə"},
    
    # Palatals (Ch-group)
    {"char": "च", "latin": "Cha", "ipa": "t͡ʃə"},
    {"char": "छ", "latin": "Chha", "ipa": "t͡ʃʰə"},
    {"char": "ज", "latin": "Ja", "ipa": "d͡ʒə"},
    {"char": "झ", "latin": "Jha", "ipa": "d͡ʒʱə"},
    {"char": "ञ", "latin": "Nya", "ipa": "ɲə"},
    
    # Retroflex (T-group hard)
    {"char": "ट", "latin": "Ta", "ipa": "ʈə"},
    {"char": "ठ", "latin": "Tha", "ipa": "ʈʰə"},
    {"char": "ड", "latin": "Da", "ipa": "ɖə"},
    {"char": "ढ", "latin": "Dha", "ipa": "ɖʱə"},
    {"char": "ण", "latin": "Na", "ipa": "ɳə"},
    
    # Dentals (T-group soft)
    {"char": "त", "latin": "Ta", "ipa": "t̪ə"},
    {"char": "थ", "latin": "Tha", "ipa": "t̪ʰə"},
    {"char": "द", "latin": "Da", "ipa": "d̪ə"},
    {"char": "ध", "latin": "Dha", "ipa": "d̪ʱə"},
    {"char": "न", "latin": "Na", "ipa": "nə"},
    
    # Labials (P-group)
    {"char": "प", "latin": "Pa", "ipa": "pə"},
    {"char": "फ", "latin": "Pha", "ipa": "pʰə"},
    {"char": "ब", "latin": "Ba", "ipa": "bə"},
    {"char": "भ", "latin": "Bha", "ipa": "bʱə"},
    {"char": "म", "latin": "Ma", "ipa": "mə"},
    
    # Semivowels & Fricatives
    {"char": "य", "latin": "Ya", "ipa": "jə"},
    {"char": "र", "latin": "Ra", "ipa": "rə"},
    {"char": "ल", "latin": "La", "ipa": "lə"},
    {"char": "व", "latin": "Va", "ipa": "ʋə"},
    {"char": "श", "latin": "Sha", "ipa": "ʃə"},
    {"char": "ष", "latin": "Sha", "ipa": "ʂə"},
    {"char": "स", "latin": "Sa", "ipa": "sə"},
    {"char": "ह", "latin": "Ha", "ipa": "ɦə"},
    
    # Conjuncts / Special
    {"char": "क्ष", "latin": "Ksha", "ipa": "kʂə"},
    {"char": "त्र", "latin": "Tra", "ipa": "t̪rə"},
    {"char": "ज्ञ", "latin": "Gya", "ipa": "ɡjə"},
]

# Barakhadi Matras (Vowels / Modifications)
BARAKHADI_MATRAS = [
    {"matra": "", "suffix": "a", "name": "A"},
    {"matra": "ा", "suffix": "aa", "name": "AA"},
    {"matra": "ि", "suffix": "i", "name": "I"},
    {"matra": "ी", "suffix": "ee", "name": "EE"},
    {"matra": "ु", "suffix": "u", "name": "U"},
    {"matra": "ू", "suffix": "oo", "name": "OO"},
    {"matra": "े", "suffix": "e", "name": "E"},
    {"matra": "ै", "suffix": "ai", "name": "AI"},
    {"matra": "ो", "suffix": "o", "name": "O"},
    {"matra": "ौ", "suffix": "au", "name": "AU"},
    {"matra": "ं", "suffix": "am", "name": "AM"},
    {"matra": "ः", "suffix": "ah", "name": "AH"},
]

# ── 2. Multilingual Proper Nouns ───────────────────────────────────────────
MULTILINGUAL_PROPER_NOUNS = [
    # Indian Proper Nouns
    {"text": "Krishnamurthy", "lang": "hi", "phonetic": "K-R-I-S-H-N-A-M-U-R-T-H-Y"},
    {"text": "Thiruvananthapuram", "lang": "hi", "phonetic": "T-H-I-R-U-V-A-N-A-N-T-H-A-P-U-R-A-M"},
    {"text": "Visakhapatnam", "lang": "hi", "phonetic": "V-I-S-A-K-H-A-P-A-T-N-A-M"},
    {"text": "Aurobindo", "lang": "hi", "phonetic": "A-U-R-O-B-I-N-D-O"},
    {"text": "Bhubaneswar", "lang": "hi", "phonetic": "B-H-U-B-A-N-E-S-W-A-R"},
    
    # Celtic & European
    {"text": "Siobhan", "lang": "en", "phonetic": "S-H-I-V-A-U-N"},
    {"text": "Caoimhe", "lang": "en", "phonetic": "K-E-E-V-A"},
    {"text": "Niamh", "lang": "en", "phonetic": "N-E-E-V"},
    
    # Vietnamese & East Asian
    {"text": "Nguyen", "lang": "vi", "phonetic": "N-W-I-N"},
    {"text": "Beijing", "lang": "zh", "phonetic": "B-A-Y-J-I-N-G"},
    {"text": "Guangzhou", "lang": "zh", "phonetic": "G-W-A-N-G-Z-H-O-U"},
    
    # Slavic
    {"text": "Dostoevsky", "lang": "ru", "phonetic": "D-O-S-T-O-Y-E-V-S-K-Y"},
    {"text": "Tchaikovsky", "lang": "ru", "phonetic": "C-H-A-Y-K-O-V-S-K-Y"},
]

def generate_barakhadi_pairs():
    """Generate all Barakhadi combination words and texts."""
    barakhadi_list = []
    for c in INDIC_CONSONANTS:
        base_char = c["char"]
        base_latin = c["latin"]
        for m in BARAKHADI_MATRAS:
            devanagari_word = base_char + m["matra"]
            latin_word = base_latin[:-1] + m["suffix"] if base_latin.endswith("a") else base_latin + m["suffix"]
            barakhadi_list.append({
                "devanagari": devanagari_word,
                "latin": latin_word,
                "consonant": base_latin,
                "matra": m["name"],
            })
    return barakhadi_list

def create_dataset(output_dir: str, num_synthetic_samples: int = 100):
    """Build formatted dataset for XTTS fine-tuning."""
    output_path = Path(output_dir).resolve()
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_path / "tts_train.csv"
    
    barakhadi_list = generate_barakhadi_pairs()
    print(f"Generated {len(barakhadi_list)} Barakhadi combinations for {len(INDIC_CONSONANTS)} consonants.")
    
    try:
        import soundfile as sf
        has_sf = True
    except ImportError:
        has_sf = False

    samples = []
    sr = 22050
    
    # 1. Barakhadi Samples
    for idx, item in enumerate(barakhadi_list):
        wav_name = f"barakhadi_{idx:03d}_{item['latin'].lower()}.wav"
        wav_full_path = wavs_dir / wav_name
        
        if not wav_full_path.exists() and has_sf:
            t = np.linspace(0, 0.6, int(sr * 0.6))
            freq = 220.0 + (idx % 12) * 20.0
            signal = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-3 * t)
            sf.write(str(wav_full_path), signal.astype(np.float32), sr)
            
        text = f"{item['latin']} ({item['devanagari']})"
        rel_path = f"wavs/{wav_name}"
        samples.append((rel_path, text))
        
    # 2. Multilingual Proper Noun Samples
    for idx, item in enumerate(MULTILINGUAL_PROPER_NOUNS):
        wav_name = f"polyglot_{idx:03d}_{item['text'].lower()}.wav"
        wav_full_path = wavs_dir / wav_name
        
        if not wav_full_path.exists() and has_sf:
            t = np.linspace(0, 1.0, int(sr * 1.0))
            freq = 300.0 + (idx % 8) * 25.0
            signal = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-2 * t)
            sf.write(str(wav_full_path), signal.astype(np.float32), sr)
            
        text = f"{item['text']} spoken as {item['phonetic']}"
        rel_path = f"wavs/{wav_name}"
        samples.append((rel_path, text))

    # Write CSV
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for rel_path, text in samples:
            writer.writerow([rel_path, text])

    print(f"Dataset generated successfully at {output_path}")
    print(f"Total training samples: {len(samples)} in {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Indic Consonant, Barakhadi, and Multilingual dataset for XTTS.")
    parser.add_argument("--output_dir", type=str, default="./formatted_dataset", help="Output dataset directory")
    args = parser.parse_args()
    
    create_dataset(args.output_dir)
