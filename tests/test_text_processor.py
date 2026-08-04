"""
Unit tests for SpeechTextProcessor, DurationPlanner, LanguageRouter, and PronunciationLexicon.
"""

import pytest
from flowedit.text.speech_input_processor import SpeechTextProcessor, ProcessedSpeechText
from flowedit.text.duration_planner import DurationPlanner
from flowedit.text.language_router import LanguageRouter
from flowedit.text.pronunciation_lexicon import PronunciationLexicon


def test_speech_text_processor_normalization_and_entities():
    processor = SpeechTextProcessor(default_language="en")
    text = "Hello, my name is Mrunmayee Sakharwade and my email is test@example.com."
    processed = processor.process(text, language="en")

    assert isinstance(processed, ProcessedSpeechText)
    assert processed.normalized_text == text
    assert len(processed.entities) >= 1

    # Ensure entity names are not split in chunks
    long_text = (
        "This is a long preliminary sentence to fill up space. "
        "Please welcome Mrunmayee Sakharwade to the conference stage today for the presentation."
    )
    chunks = processor.chunk_text_entity_safe(long_text, processed.entities, max_chunk_len=60)
    for chunk in chunks:
        # Assert "Mrunmayee" and "Sakharwade" stay in the same chunk if present
        if "Mrunmayee" in chunk:
            assert "Sakharwade" in chunk


def test_duration_planner_dynamic_speed():
    planner = DurationPlanner(speed_floor=0.82, speed_ceiling=1.02)
    
    # Short target text with long reference audio -> should slow down (speed closer to floor)
    plan_slow = planner.plan_duration(
        target_text="Hello world.",
        ref_audio_duration=5.0,
        ref_text="This is a very long reference sentence read slowly.",
    )
    assert 0.82 <= plan_slow.dynamic_speed_factor <= 1.02

    # Fast reference rate -> should clamp within bounds
    plan_fast = planner.plan_duration(
        target_text="This is a comprehensive test sentence with multiple target words.",
        ref_audio_duration=1.0,
        ref_text="Quick test.",
    )
    assert 0.82 <= plan_fast.dynamic_speed_factor <= 1.02


def test_language_router():
    router = LanguageRouter()
    
    # Same language prompt and target -> zero_shot
    route1 = router.route(prompt_language="en", target_language="en")
    assert route1.mode == "zero_shot"

    # Different prompt and target -> cross_lingual
    route2 = router.route(prompt_language="zh", target_language="en")
    assert route2.mode == "cross_lingual"

    # Non-Latin native script entity
    route3 = router.route(
        prompt_language="en",
        target_language="en",
        entity_text="मृण्मयी साखरवाडे",
    )
    assert route3.requires_phonetic_alias is True
    assert route3.entity_language == "hi-IN"


def test_pronunciation_lexicon():
    lexicon = PronunciationLexicon()
    alias = lexicon.get_alias("Mrunmayee Sakharwade")

    assert alias is not None
    assert alias.native_script_alias == "मृण्मयी साखरवाडे"

    rendered = lexicon.render_text_with_alias(
        "Welcome Mrunmayee Sakharwade to the event.", mode="native_script"
    )
    assert "मृण्मयी साखरवाडे" in rendered
