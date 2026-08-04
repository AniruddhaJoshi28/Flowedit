"""
Pronunciation Lexicon for FlowEdit.

Maintains phonetic and native-script aliases (e.g. mapping "Mrunmayee Sakharwade"
to Devanagari script "मृण्मयी साखरवाडे") without permanently mutating canonical text.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PronunciationAlias:
    canonical: str
    phonetic_alias: Optional[str] = None
    native_script_alias: Optional[str] = None
    language: str = "en"


class PronunciationLexicon:
    """Manages phonetic and native script aliases for target entities."""

    def __init__(self):
        self._entries: Dict[str, PronunciationAlias] = {}
        # Pre-seed known difficult entity aliases
        self.add_alias(
            canonical="Mrunmayee Sakharwade",
            native_script_alias="मृण्मयी साखरवाडे",
            phonetic_alias="Mroon-ma-yee Sa-khar-wa-day",
            language="mr-IN",
        )

    def add_alias(
        self,
        canonical: str,
        native_script_alias: Optional[str] = None,
        phonetic_alias: Optional[str] = None,
        language: str = "en",
    ):
        key = canonical.lower().strip()
        self._entries[key] = PronunciationAlias(
            canonical=canonical,
            native_script_alias=native_script_alias,
            phonetic_alias=phonetic_alias,
            language=language,
        )

    def get_alias(self, text: str) -> Optional[PronunciationAlias]:
        key = text.lower().strip()
        return self._entries.get(key)

    def render_text_with_alias(self, text: str, mode: str = "native_script") -> str:
        """Returns rendered text substitution for backend synthesis without mutating canonical source."""
        rendered = text
        for key, entry in self._entries.items():
            if entry.canonical.lower() in rendered.lower():
                alias_str = None
                if mode == "native_script" and entry.native_script_alias:
                    alias_str = entry.native_script_alias
                elif mode == "phonetic" and entry.phonetic_alias:
                    alias_str = entry.phonetic_alias

                if alias_str:
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(entry.canonical), re.IGNORECASE)
                    rendered = pattern.sub(alias_str, rendered)
        return rendered
