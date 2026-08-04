"""
Shared Speech Text Processor for FlowEdit.

Implements text normalization, entity extraction, and entity-safe chunking.
Ensures names, numbers, email addresses, and phonetic aliases are never split.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class PronunciationEntity:
    original_text: str
    normalized_text: str
    phonetic_alias: Optional[str] = None
    native_script_alias: Optional[str] = None
    start_char_idx: int = -1
    end_char_idx: int = -1
    language: Optional[str] = None


@dataclass
class ProcessedSpeechText:
    original_text: str
    normalized_text: str
    language: str
    segments: List[str]
    entities: List[PronunciationEntity]
    estimated_syllables: int
    punctuation_pause_seconds: float


class SpeechTextProcessor:
    """Normalizes input text and chunks sentences while preserving named entities."""

    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        # Entity detection patterns (Names, acronyms, decimals, phone numbers, emails, dates)
        self.entity_pattern = re.compile(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)|'     # Proper names (e.g., Mrunmayee Sakharwade)
            r'(\b[A-Z]{2,}\b)|'                       # Acronyms (e.g., NASA, AI)
            r'(\b\d+(?:\.\d+)?\b)|'                   # Decimals / numbers
            r'(\b[\w\.-]+@[\w\.-]+\.\w+\b)|'          # Email addresses
            r'(\+?\d[\d\s-]{7,}\d)'                   # Phone numbers
        )

    def process(
        self,
        text: str,
        language: Optional[str] = None,
        entity_hints: Optional[List[str]] = None,
    ) -> ProcessedSpeechText:
        """Processes text through NFC normalization, entity extraction, and entity-safe chunking."""
        lang = language or self.default_language

        # Step 1: Unicode NFC normalization & whitespace cleaning
        norm_text = unicodedata.normalize("NFC", text.strip())
        norm_text = re.sub(r'\s+', ' ', norm_text)

        # Step 2: Extract entities
        entities = self._extract_entities(norm_text, entity_hints=entity_hints, language=lang)

        # Step 3: Estimate syllables
        syllables = self.estimate_syllables(norm_text)

        # Step 4: Estimate punctuation pause duration
        pause_sec = self._calculate_punctuation_pauses(norm_text)

        # Step 5: Entity-safe sentence chunking
        segments = self.chunk_text_entity_safe(norm_text, entities)

        return ProcessedSpeechText(
            original_text=text,
            normalized_text=norm_text,
            language=lang,
            segments=segments,
            entities=entities,
            estimated_syllables=syllables,
            punctuation_pause_seconds=pause_sec,
        )

    def _extract_entities(
        self, text: str, entity_hints: Optional[List[str]] = None, language: str = "en"
    ) -> List[PronunciationEntity]:
        entities = []

        if entity_hints:
            for hint in entity_hints:
                for match in re.finditer(re.escape(hint), text, re.IGNORECASE):
                    entities.append(
                        PronunciationEntity(
                            original_text=match.group(0),
                            normalized_text=match.group(0),
                            start_char_idx=match.start(),
                            end_char_idx=match.end(),
                            language=language,
                        )
                    )

        for match in self.entity_pattern.finditer(text):
            matched_str = match.group(0)
            if not any(e.start_char_idx == match.start() for e in entities):
                entities.append(
                    PronunciationEntity(
                        original_text=matched_str,
                        normalized_text=matched_str,
                        start_char_idx=match.start(),
                        end_char_idx=match.end(),
                        language=language,
                    )
                )

        return entities

    def chunk_text_entity_safe(
        self, text: str, entities: List[PronunciationEntity], max_chunk_len: int = 150
    ) -> List[str]:
        """Splits long text into sentence chunks without breaking entity boundaries."""
        # Initial sentence splits on punctuation
        sentence_spans = []
        for match in re.finditer(r'[^.!?]+[.!?]*', text):
            sentence_spans.append((match.start(), match.end(), match.group(0).strip()))

        chunks = []
        current_chunk = ""

        for start, end, s_text in sentence_spans:
            # Check if splitting at 'end' would cut inside any entity
            overlaps = any(e.start_char_idx < end < e.end_char_idx for e in entities)
            
            if overlaps or (len(current_chunk) + len(s_text) <= max_chunk_len):
                current_chunk = (current_chunk + " " + s_text).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = s_text

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    @staticmethod
    def estimate_syllables(text: str) -> int:
        """Estimates syllable count for pacing calculations."""
        words = re.findall(r'\b\w+\b', text.lower())
        count = 0
        for word in words:
            # Basic vowel count heuristic
            vowels = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e') and len(word) > 2 and not word.endswith('le'):
                vowels = max(1, vowels - 1)
            count += max(1, vowels)
        return max(1, count)

    @staticmethod
    def _calculate_punctuation_pauses(text: str) -> float:
        """Calculates total pause seconds from commas, colons, and sentence breaks."""
        commas = len(re.findall(r'[,;:]', text))
        periods = len(re.findall(r'[.!?]', text))
        return (commas * 0.15) + (periods * 0.30)
