"""
Language Router for FlowEdit.

Routes synthesis requests to zero-shot or cross-lingual modes.
Supports separate sentence_language and entity_language to handle
non-Latin named entities embedded within English sentences.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LanguageRoute:
    sentence_language: str
    entity_language: Optional[str]
    mode: str                    # "zero_shot" or "cross_lingual"
    requires_phonetic_alias: bool


class LanguageRouter:
    """Routes text and speaker prompt language configurations."""

    def route(
        self,
        prompt_language: str,
        target_language: str,
        user_hint_language: Optional[str] = None,
        entity_text: Optional[str] = None,
    ) -> LanguageRoute:
        sentence_lang = user_hint_language or target_language or "en"
        
        # Determine entity-specific language (e.g. Marathi/Hindi names in Latin script)
        entity_lang = None
        requires_alias = False

        if entity_text:
            # Check for non-ASCII scripts (Devanagari, etc.)
            has_devanagari = any("\u0900" <= char <= "\u097F" for char in entity_text)
            if has_devanagari:
                entity_lang = "hi-IN"
                requires_alias = True
            elif prompt_language.startswith("mr") or prompt_language.startswith("hi"):
                entity_lang = prompt_language

        # Zero-shot mode applies when prompt language matches sentence target language
        if prompt_language.split("-")[0].lower() == sentence_lang.split("-")[0].lower():
            mode = "zero_shot"
        else:
            mode = "cross_lingual"

        return LanguageRoute(
            sentence_language=sentence_lang,
            entity_language=entity_lang,
            mode=mode,
            requires_phonetic_alias=requires_alias,
        )
