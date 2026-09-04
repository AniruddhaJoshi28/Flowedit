"""
Whisper Forced Alignment — Stage 1 of FlowEdit.

Paper Section 3.2 (Stage 1: Detection and Grounding):
    "The user provides a corrective signal: a reference audio y_ref
    paired with the target text. We use Whisper-Large-v3 forced alignment
    to localize the temporal boundaries of the target word, extracting
    the target token indices I. We expand this by one token on each side
    to absorb tokenizer boundary errors."

This module:
    1. Takes reference audio + target text
    2. Runs Whisper forced alignment to find word boundaries
    3. Maps word boundaries to XTTS-2 token indices
    4. Returns token indices with ±1 padding
"""

import torch
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from flowedit.config import AlignmentConfig

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of forced alignment for a target word.

    Attributes:
        token_indices: List of XTTS-2 token indices covering the target word
                       (expanded by ±1 token per paper specification)
        start_time: Start time of the target word in seconds
        end_time: End time of the target word in seconds
        confidence: Alignment confidence score
        word: The matched word text
        full_transcript: Full whisper transcript for verification
        char_start: Character start index of matched word in text
        char_end: Character end index of matched word in text
    """
    token_indices: List[int]
    start_time: float
    end_time: float
    confidence: float
    word: str
    full_transcript: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    auto_phonetic_hint: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class WhisperAligner:
    """Whisper-based forced alignment and transcription for FlowEdit Stage 1 and STT.

    Uses Whisper (via whisperx or openai-whisper) to find temporal boundaries
    of target words in reference audio, or transcribe free speech to text.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        self._model = None
        self._backend = None

    def load_model(self) -> None:
        """Load the Whisper model for alignment and speech-to-text."""
        import torch

        logger.info(f"Loading Whisper model: {self.config.whisper_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self._backend = None
        
        try:
            import whisperx
            self._model = whisperx.load_model(
                self.config.whisper_model, 
                device=device, 
                compute_type=compute_type
            )
            self._backend = "whisperx"
            logger.info("Whisper model (whisperx) loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Whisper model '{self.config.whisper_model}' ({e}). Attempting fallback to 'base'...")
            try:
                import whisperx
                self._model = whisperx.load_model(
                    "base",
                    device=device,
                    compute_type=compute_type
                )
                self._backend = "whisperx"
                logger.info("Whisper 'base' model loaded successfully as fallback.")
            except Exception as e2:
                logger.warning(f"Failed to load whisperx fallback ({e2}). Attempting openai-whisper...")
                try:
                    import whisper
                    self._model = whisper.load_model("base", device=device)
                    self._backend = "whisper"
                    logger.info("OpenAI Whisper 'base' loaded successfully.")
                except Exception as e3:
                    logger.warning(f"Failed to load Whisper model ({e3}). Whisper alignment will operate in fallback mode.")
                    self._model = None
                    self._backend = None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        return_timestamps: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe speech in an audio file to text using Whisper.

        Args:
            audio_path: Path to the audio file (wav, mp3, flac, etc.)
            language: Optional language code (e.g., 'en', 'hi', 'fr') or None/'auto' to auto-detect
            return_timestamps: Whether to include segment-level timestamps in the response
            initial_prompt: Optional biasing prompt (e.g. correct vocabulary from Hopfield Memory)

        Returns:
            Dict containing:
                - text: Complete transcribed text
                - language: Detected or specified language
                - duration: Audio duration in seconds
                - segments: List of segment dicts with start, end, text
        """
        self._ensure_loaded()
        audio_path = str(Path(audio_path).resolve())
        lang = None if (not language or language.lower() in ("auto", "none", "")) else language.lower()

        # Calculate duration
        duration = 0.0
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = round(float(info.duration), 3)
        except Exception:
            try:
                import librosa
                duration = round(float(librosa.get_duration(path=audio_path)), 3)
            except Exception:
                pass

        if self._model is None:
            raise RuntimeError(
                "Whisper model is not available. Please ensure 'whisperx' or 'openai-whisper' is installed."
            )

        raw_segments = []
        detected_language = lang or "unknown"
        full_text = ""

        try:
            if getattr(self, "_backend", None) == "whisper":
                transcribe_kwargs = {}
                if lang:
                    transcribe_kwargs["language"] = lang
                if initial_prompt:
                    transcribe_kwargs["initial_prompt"] = initial_prompt
                res = self._model.transcribe(audio_path, **transcribe_kwargs)
                full_text = res.get("text", "").strip()
                detected_language = res.get("language", lang or "unknown")
                raw_segments = res.get("segments", [])
            else:
                import whisperx
                audio = whisperx.load_audio(audio_path)
                transcribe_kwargs = {"language": lang}
                if initial_prompt:
                    transcribe_kwargs["initial_prompt"] = initial_prompt
                try:
                    res = self._model.transcribe(audio, **transcribe_kwargs)
                except (TypeError, ValueError):
                    # In case underlying faster-whisper version does not support initial_prompt
                    res = self._model.transcribe(audio, language=lang)
                raw_segments = res.get("segments", [])
                detected_language = res.get("language", lang or "unknown")
                full_text = " ".join(s.get("text", "").strip() for s in raw_segments).strip()
                if not full_text and "text" in res:
                    full_text = res["text"].strip()
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise e

        formatted_segments = []
        if return_timestamps:
            for i, seg in enumerate(raw_segments):
                seg_text = seg.get("text", "").strip()
                if seg_text:
                    seg_start = round(float(seg.get("start", 0.0)), 3)
                    seg_end = round(float(seg.get("end", 0.0)), 3)
                    words = []
                    raw_words = seg.get("words", [])
                    if raw_words:
                        for w in raw_words:
                            w_str = (w.get("word", "") if isinstance(w, dict) else str(w)).strip()
                            if w_str:
                                words.append({
                                    "word": w_str,
                                    "start": round(float(w.get("start", seg_start) if isinstance(w, dict) else seg_start), 3),
                                    "end": round(float(w.get("end", seg_end) if isinstance(w, dict) else seg_end), 3),
                                })
                    if not words and seg_text:
                        tokens = seg_text.split()
                        dur = max(seg_end - seg_start, 0.01)
                        step = dur / max(len(tokens), 1)
                        for w_i, tok in enumerate(tokens):
                            words.append({
                                "word": tok,
                                "start": round(seg_start + w_i * step, 3),
                                "end": round(seg_start + (w_i + 1) * step, 3),
                            })
                    formatted_segments.append({
                        "id": i,
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_text,
                        "words": words,
                    })

        return {
            "text": full_text,
            "language": detected_language,
            "duration": duration,
            "segments": formatted_segments if return_timestamps else [],
        }

    def transcribe_stream(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ):
        """Transcribe speech in an audio file, yielding progressive segment events in real time.

        Args:
            audio_path: Path to the audio file
            language: Optional language code or None/'auto'
            initial_prompt: Optional biasing prompt from Hopfield Memory

        Yields:
            Dict events:
                - {"type": "metadata", "language": str, "duration": float}
                - {"type": "segment", "id": int, "start": float, "end": float, "text": str, "partial_transcript": str}
                - {"type": "complete", "text": str, "language": str, "duration": float, "segments": list}
        """
        self._ensure_loaded()
        audio_path = str(Path(audio_path).resolve())
        lang = None if (not language or language.lower() in ("auto", "none", "")) else language.lower()

        # Calculate duration
        duration = 0.0
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = round(float(info.duration), 3)
        except Exception:
            try:
                import librosa
                duration = round(float(librosa.get_duration(path=audio_path)), 3)
            except Exception:
                pass

        if self._model is None:
            raise RuntimeError("Whisper model is not loaded.")

        # Emit initial metadata
        yield {
            "type": "metadata",
            "language": lang or "auto",
            "duration": duration,
        }

        # Transcribe with timestamps
        result = self.transcribe(
            audio_path=audio_path,
            language=language,
            return_timestamps=True,
            initial_prompt=initial_prompt,
        )

        detected_lang = result.get("language", lang or "unknown")
        segments = result.get("segments", [])

        # Stream out segments progressively
        accumulated_texts = []
        for i, seg in enumerate(segments):
            seg_text = seg.get("text", "").strip()
            if seg_text:
                accumulated_texts.append(seg_text)
                yield {
                    "type": "segment",
                    "id": seg.get("id", i),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg_text,
                    "words": seg.get("words", []),
                    "partial_transcript": " ".join(accumulated_texts),
                }

        yield {
            "type": "complete",
            "text": result.get("text", ""),
            "language": detected_lang,
            "duration": duration,
            "segments": segments,
        }


    def align(
        self,
        audio_path: str,
        target_word: str,
        full_text: Optional[str] = None,
        language: Optional[str] = None,
        ref_is_word_only: bool = False,
    ) -> AlignmentResult:
        """Perform forced alignment to find target word boundaries.

        This implements Paper Stage 1: Detection and Grounding.

        Args:
            audio_path: Path to reference audio containing correct pronunciation
            target_word: The word to locate (e.g., "Siobhan")
            full_text: Optional full text for better alignment context.
                       If provided, Whisper aligns to this text.
                       If None, Whisper transcribes freely.
            language: Language hint (None = auto-detect)
            ref_is_word_only: If True, the reference audio contains ONLY
                             the target word. Use free transcription to get
                             timing, and treat the entire audio as the word.

        Returns:
            AlignmentResult with word boundaries and token indices

        Raises:
            ValueError: If target word is not found in the audio
        """
        self._ensure_loaded()
        import whisperx
        import torch

        audio_path = str(Path(audio_path).resolve())
        language = language or self.config.language
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Run Whisper with word-level timestamps
        logger.info(f"Aligning '{target_word}' in {audio_path}")
        
        # Load audio using whisperx
        audio = whisperx.load_audio(audio_path)
        
        # 1. Handle Isolated Word Fallback via VAD + Auto-Phonetic Transcription
        if ref_is_word_only:
            import librosa
            import numpy as np
            logger.info(f"Using Audio-Boundary VAD + Whisper acoustic transcription for target_word: '{target_word}'")
            
            # Load audio for boundary detection
            y, sr = librosa.load(audio_path, sr=16000)
            duration = float(len(y)) / sr
            
            # Use librosa.effects.split with generous threshold to avoid clipping plosives/fricatives
            non_silent_intervals = librosa.effects.split(y, top_db=35)
            if len(non_silent_intervals) > 0:
                start_sample = non_silent_intervals[0][0]
                end_sample = non_silent_intervals[-1][1]
                start_time = max(0.0, float(start_sample) / sr - 0.08)
                end_time = min(duration, float(end_sample) / sr + 0.08)
            else:
                start_time = 0.0
                end_time = duration

            auto_phonetic_hint = None
            try:
                if self._model is not None:
                    trans_res = self._model.transcribe(audio, language=language)
                    if "segments" in trans_res and trans_res["segments"]:
                        raw_text = " ".join(s.get("text", "") for s in trans_res["segments"]).strip()
                        cleaned_text = re.sub(r'[^\w\s-]', '', raw_text).strip()
                        if cleaned_text:
                            auto_phonetic_hint = cleaned_text
                            logger.info(f"✓ Whisper acoustic transcription of ref audio: '{auto_phonetic_hint}'")
            except Exception as e:
                logger.warning(f"Acoustic transcription failed ({e}). Operating in acoustic boundary mode.")

            return AlignmentResult(
                token_indices=[],  # Will be mapped to XTTS tokens in map_to_token_indices
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,
                word=target_word,
                full_transcript=auto_phonetic_hint or target_word,
                char_start=0,
                char_end=len(target_word),
                auto_phonetic_hint=auto_phonetic_hint,
            )

        # 1. Transcribe with Whisper (or use full_text directly for forced alignment)

        if full_text:
            # Bypass free transcription to guarantee target_word matches perfectly
            import librosa
            duration = librosa.get_duration(path=audio_path)
            result = {
                "segments": [{"text": full_text, "start": 0.0, "end": duration}],
                "language": language or "en"
            }
        else:
            result = self._model.transcribe(audio, language=language)
        
        # 2. Align with Wav2Vec2
        align_language = result.get("language", language or "en")
        model_a, metadata = whisperx.load_align_model(language_code=align_language, device=device)
        
        # In whisperx, align modifies the segments to include word-level timings
        try:
            aligned_result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                device, 
                return_char_alignments=False
            )
            
            # Check if forced alignment returned empty word segments (often happens if full_text contains digits/symbols not in wav2vec2 dict)
            if full_text and not self._extract_word_segments(aligned_result):
                raise ValueError("Forced alignment returned no words")
                
        except Exception as e:
            if full_text:
                logger.warning(f"Forced alignment with full_text failed ({e}). Falling back to free transcription.")
                result = self._model.transcribe(audio, language=language)
                
                # Re-align with transcribed segments
                aligned_result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    device, 
                    return_char_alignments=False
                )
            else:
                raise e

        if ref_is_word_only:
            # Reference audio contains ONLY the target word.
            words = self._extract_word_segments(aligned_result)
            full_transcript = " ".join(w["word"] for w in words)

            if words:
                # Try to find the target word in transcription
                match = self._find_target_word(words, target_word)
                if match is None:
                    match = self._fuzzy_find_target(words, target_word)
                if match is None:
                    # Whisper may transcribe the word differently
                    logger.info(
                        f"Whisper transcribed as '{full_transcript}', "
                        f"but we know the entire audio is '{target_word}'. "
                        f"Using full audio boundaries."
                    )
                    match = {
                        "word": target_word,
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "confidence": sum(w.get("confidence", 0.5) for w in words) / len(words),
                    }
            else:
                # No words detected — use full audio duration
                import librosa
                duration = librosa.get_duration(path=audio_path)
                match = {
                    "word": target_word,
                    "start": 0.0,
                    "end": duration,
                    "confidence": 0.5,
                }
                full_transcript = target_word

            logger.info(
                f"Word-only ref: '{match['word']}' at "
                f"{match['start']:.2f}s - {match['end']:.2f}s "
                f"(confidence: {match.get('confidence', 0):.2f})"
            )

            auto_hint = match["word"] if match["word"].lower() != target_word.lower() else None
            return AlignmentResult(
                token_indices=[],  # Populated by map_to_token_indices()
                start_time=match["start"],
                end_time=match["end"],
                confidence=match.get("confidence", 0.0),
                word=match["word"],
                full_transcript=full_transcript,
                auto_phonetic_hint=auto_hint
            )

        # Extract word-level segments from the aligned result
        words = self._extract_word_segments(aligned_result)

        if not words:
            raise ValueError(
                f"No words detected in audio: {audio_path}. "
                "Ensure the audio contains clear speech."
            )

        # Find the target word in the aligned output
        match = self._find_target_word(words, target_word)

        if match is None:
            # Try fuzzy matching
            match = self._fuzzy_find_target(words, target_word)

        if match is None:
            transcript = " ".join(w["word"] for w in words)
            raise ValueError(
                f"Target word '{target_word}' not found in transcript: "
                f"'{transcript}'. Check that the reference audio contains "
                f"the target word spoken clearly."
            )

        word_info = match
        if word_info.get("start", 0.0) >= word_info.get("end", 0.0):
            raise ValueError(
                f"Invalid word boundaries detected for '{target_word}': "
                f"start={word_info.get('start')}s, end={word_info.get('end')}s. "
                "Whisperx failed to resolve valid timestamps."
            )

        full_transcript = " ".join(w["word"] for w in words)

        logger.info(
            f"Found '{word_info['word']}' at "
            f"{word_info['start']:.2f}s - {word_info['end']:.2f}s "
            f"(confidence: {word_info.get('confidence', 0):.2f})"
        )

        auto_hint = word_info["word"] if word_info["word"].lower() != target_word.lower() else None
        return AlignmentResult(
            token_indices=[],  # Populated by map_to_token_indices()
            start_time=word_info["start"],
            end_time=word_info["end"],
            confidence=word_info.get("confidence", 0.0),
            word=word_info["word"],
            full_transcript=full_transcript,
            auto_phonetic_hint=auto_hint
        )

    def map_to_token_indices(
        self,
        alignment: AlignmentResult,
        full_text: str,
        target_word: str,
        tokenizer,
        language: str = "en",
        occurrence_index: int = 0,
        target_phrase: Optional[str] = None,
    ) -> AlignmentResult:
        """Map word-level alignment to XTTS-2 token indices.

        Paper: "We expand this by one token on each side to absorb
        tokenizer boundary errors."

        Args:
            alignment: AlignmentResult from align()
            full_text: The complete text being synthesized
            target_word: The target word
            tokenizer: XTTS-2 tokenizer instance
            language: Language code
            occurrence_index: 0-based occurrence index if multiple occurrences exist (supports negative indexing e.g. -1 for last)
            target_phrase: Optional surrounding phrase (e.g. "lead pipes") to uniquely locate target word occurrence

        Returns:
            Updated AlignmentResult with token_indices populated
        """
        import re

        # Tokenize the full text
        all_token_ids = tokenizer.encode(full_text, lang=language)
        if isinstance(all_token_ids, torch.Tensor):
            all_token_ids = all_token_ids.tolist()
        if isinstance(all_token_ids[0], list):
            all_token_ids = all_token_ids[0]

        # Fast path: If full_text is the isolated target word, map all content tokens
        if full_text.strip().lower() == target_word.strip().lower():
            start_tok = 1 if (hasattr(tokenizer, "bos_token_id") and all_token_ids and all_token_ids[0] == getattr(tokenizer, "bos_token_id", None)) else 0
            end_tok = len(all_token_ids) - 1 if (hasattr(tokenizer, "eos_token_id") and all_token_ids and all_token_ids[-1] == getattr(tokenizer, "eos_token_id", None)) else len(all_token_ids)
            alignment.token_indices = list(range(start_tok, max(start_tok + 1, end_tok)))
            alignment.char_start = 0
            alignment.char_end = len(full_text)
            logger.info(
                f"Mapped isolated '{target_word}' to token indices {alignment.token_indices} (total tokens: {len(all_token_ids)})"
            )
            return alignment

        # Normalize both target_word, phrase and full_text
        from flowedit.utils.indic_phonetics import normalize_indic_phonetics
        target_norm = normalize_indic_phonetics(target_word).lower()
        text_norm = normalize_indic_phonetics(full_text).lower()

        char_start = -1
        char_end = -1

        # 1. Target phrase grounding (e.g. user entered "lead pipes")
        if target_phrase:
            phrase_norm = normalize_indic_phonetics(target_phrase).lower()
            phrase_matches = [m.span() for m in re.finditer(re.escape(phrase_norm), text_norm)]
            if phrase_matches:
                p_start, p_end = phrase_matches[min(max(0, occurrence_index), len(phrase_matches) - 1)]
                phrase_substr = text_norm[p_start:p_end]
                w_match = re.search(r'\b' + re.escape(target_norm) + r'\b', phrase_substr) or re.search(re.escape(target_norm), phrase_substr)
                if w_match:
                    char_start = p_start + w_match.start()
                    char_end = p_start + w_match.end()

        # 2. Primary search: word boundary occurrences
        if char_start == -1:
            wb_matches = [m.span() for m in re.finditer(r'\b' + re.escape(target_norm) + r'\b', text_norm)]
            if not wb_matches:
                # Substring occurrences
                wb_matches = [m.span() for m in re.finditer(re.escape(target_norm), text_norm)]

            if wb_matches:
                if occurrence_index < 0:
                    chosen_idx = max(0, len(wb_matches) + occurrence_index)
                else:
                    chosen_idx = min(occurrence_index, len(wb_matches) - 1)
                char_start, char_end = wb_matches[chosen_idx]
            else:
                # Secondary search: clean alphanumeric matching
                target_clean = self._normalize_word(target_word)
                text_clean = self._normalize_word(full_text)
                clean_matches = [m.span() for m in re.finditer(re.escape(target_clean), text_clean)]
                if clean_matches:
                    if occurrence_index < 0:
                        chosen_idx = max(0, len(clean_matches) + occurrence_index)
                    else:
                        chosen_idx = min(occurrence_index, len(clean_matches) - 1)
                    char_start, char_end = clean_matches[chosen_idx]


        if char_start == -1:
            # Tertiary search: partial prefix match
            for i in range(len(text_norm)):
                if text_norm[i:].startswith(target_norm[:3]):
                    char_start = i
                    char_end = char_start + len(target_norm)
                    break

        if char_start == -1:
            logger.warning(
                f"Could not find '{target_word}' (norm: '{target_norm}') in text '{full_text}'. "
                f"Bounding target token span."
            )
            total_tokens = len(all_token_ids)
            target_ratio = max(0.1, len(target_word) / max(1, len(full_text)))
            num_target_tokens = max(2, int(total_tokens * target_ratio))
            mid = total_tokens // 2
            token_indices = list(range(
                max(0, mid - num_target_tokens // 2),
                min(total_tokens, mid + (num_target_tokens + 1) // 2)
            ))
            alignment.char_start = None
            alignment.char_end = None
        else:
            # Map character positions to token positions
            token_indices = self._chars_to_token_indices(
                all_token_ids, tokenizer, full_text,
                char_start, char_end, language
            )
            alignment.char_start = char_start
            alignment.char_end = char_end

        # Expand by ±1 token (paper specification)
        expand = self.config.token_expand
        if token_indices:
            min_idx = max(0, min(token_indices) - expand)
            max_idx = min(len(all_token_ids) - 1, max(token_indices) + expand)
            token_indices = list(range(min_idx, max_idx + 1))

        # Update alignment result
        alignment.token_indices = token_indices

        logger.info(
            f"Mapped '{target_word}' to token indices {token_indices} "
            f"(total tokens: {len(all_token_ids)}, "
            f"expand ±{expand})"
        )

        return alignment

    def _extract_word_segments(self, result) -> List[dict]:
        """Extract word-level segments from WhisperX output.

        Handles different output formats from whisperx and stable-ts.
        """
        words = []

        if hasattr(result, "segments"):
            for segment in result.segments:
                if hasattr(segment, "words"):
                    for word in segment.words:
                        words.append({
                            "word": word.word.strip() if hasattr(word, "word") else str(word).strip(),
                            "start": word.start if hasattr(word, "start") else 0,
                            "end": word.end if hasattr(word, "end") else 0,
                            "confidence": getattr(word, "score", getattr(word, "probability", 0.0)),
                        })
        elif isinstance(result, dict) and "segments" in result:
            for segment in result["segments"]:
                for word in segment.get("words", []):
                    words.append({
                        "word": word.get("word", "").strip(),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "confidence": word.get("score", word.get("probability", 0.0)),
                    })

        return words

    def _find_target_word(
        self,
        words: List[dict],
        target: str,
    ) -> Optional[dict]:
        """Find exact match for target word in aligned words."""
        target_clean = self._normalize_word(target)

        for word_info in words:
            word_clean = self._normalize_word(word_info["word"])
            if word_clean == target_clean:
                return word_info

        return None

    def _fuzzy_find_target(
        self,
        words: List[dict],
        target: str,
    ) -> Optional[dict]:
        """Fuzzy match target word in aligned words.

        Handles cases where Whisper transcribes the word differently
        (e.g., "Siobhan" might be transcribed as "Shavon").
        Also handles when Whisper splits one word into multiple (e.g., "Mrunmayee" -> "Munroon May").
        """
        target_clean = self._normalize_word(target)

        best_match = None
        best_score = 0.0

        # Check single words and combinations of up to 4 adjacent words
        for window_size in range(1, min(5, len(words) + 1)):
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                combined_word = "".join(w["word"] for w in window)
                combined_clean = self._normalize_word(combined_word)
                
                score = self._similarity_score(combined_clean, target_clean)
                
                # Boost score if phonetic normalizations match
                try:
                    from flowedit.utils.indic_phonetics import normalize_indic_phonetics
                    phonetic_combined = self._normalize_word(normalize_indic_phonetics(combined_word))
                    phonetic_target = self._normalize_word(normalize_indic_phonetics(target))
                    phonetic_score = self._similarity_score(phonetic_combined, phonetic_target)
                    score = max(score, phonetic_score)
                except ImportError:
                    pass

                # Length coverage weight to prevent partial substrings (like 'specific') from beating fuller phrases (like 'buy specific')
                if combined_clean and target_clean:
                    coverage = min(len(combined_clean), len(target_clean)) / max(len(combined_clean), len(target_clean))
                    adjusted_score = score * (0.6 + 0.4 * coverage)
                else:
                    adjusted_score = score
                
                if adjusted_score > best_score and adjusted_score > 0.4:
                    best_score = adjusted_score
                    # Create a merged word_info dictionary spanning the window
                    best_match = {
                        "word": " ".join(w["word"] for w in window),
                        "start": window[0]["start"],
                        "end": window[-1]["end"],
                        "confidence": sum(w.get("confidence", 0.0) for w in window) / len(window)
                    }

        if best_match:
            logger.info(
                f"Fuzzy matched '{target}' → '{best_match['word']}' "
                f"(score: {best_score:.2f})"
            )

        return best_match

    def _chars_to_token_indices(
        self,
        token_ids: List[int],
        tokenizer,
        full_text: str,
        char_start: int,
        char_end: int,
        language: str,
    ) -> List[int]:
        """Map character span to token indices.

        This handles the mismatch between character positions and
        BPE/SentencePiece token boundaries.
        
        For F5-TTS character-level tokenizers, each character maps
        directly to one token (1:1 mapping).
        """
        # Fast path: F5-TTS uses character-level tokenization
        # Each character = exactly 1 token, so char index == token index
        if len(token_ids) == len(full_text):
            token_indices = list(range(
                max(0, char_start),
                min(len(token_ids), char_end)
            ))
            if token_indices:
                return token_indices

        token_indices = []

        # Try prefix decoding to map char positions to token positions
        if hasattr(tokenizer, 'decode'):
            decoded_prefixes = []
            for i in range(1, len(token_ids) + 1):
                try:
                    prefix = tokenizer.decode(token_ids[:i])
                    decoded_prefixes.append(len(prefix))
                except Exception:
                    decoded_prefixes.append(decoded_prefixes[-1] if decoded_prefixes else 0)

            prev_len = 0
            for idx, current_len in enumerate(decoded_prefixes):
                token_start = prev_len
                token_end = current_len

                # Check if this token overlaps with our target span
                if token_end > char_start and token_start < char_end:
                    token_indices.append(idx)

                prev_len = current_len

        if not token_indices:
            # Fallback: estimate based on character ratio
            total_chars = len(full_text)
            total_tokens = len(token_ids)
            ratio = total_tokens / max(total_chars, 1)

            est_start = int(char_start * ratio)
            est_end = int(char_end * ratio) + 1

            token_indices = list(range(
                max(0, est_start),
                min(total_tokens, est_end)
            ))

        return token_indices

    @staticmethod
    def _normalize_word(word: str) -> str:
        """Normalize a word for comparison."""
        return re.sub(r'[^\w]', '', word.lower().strip())

    @staticmethod
    def _similarity_score(a: str, b: str) -> float:
        """Simple character-level similarity score."""
        if not a or not b:
            return 0.0

        # Character bigram overlap (Dice coefficient)
        def bigrams(s):
            return set(s[i:i + 2] for i in range(len(s) - 1))

        bg_a = bigrams(a)
        bg_b = bigrams(b)

        if not bg_a or not bg_b:
            return 1.0 if a == b else 0.0

        overlap = len(bg_a & bg_b)
        return 2 * overlap / (len(bg_a) + len(bg_b))

    def _ensure_loaded(self) -> None:
        """Ensure the Whisper model is loaded."""
        if self._model is None:
            self.load_model()
