"""
Candidate Selector & 2-Tier Fast/Slow Evaluator for FlowEdit.

Implements staged candidate evaluation:
- Fast validation path on normal synthesis (audio decodability, duration, silence, clipping, repetition).
- Slow validation path (Whisper WER/CER, phoneme alignment, speaker similarity) triggered ONLY during /api/correct, memory updates, fast-path failure, or benchmark mode.
"""

from dataclasses import dataclass
import torch
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple

from flowedit.evaluation.audio_artifact_detector import AudioArtifactDetector, ArtifactAssessment
from flowedit.text.duration_planner import DurationPlanner

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    candidate_index: int
    waveform: torch.Tensor
    sample_rate: int
    assessment: ArtifactAssessment
    fast_path_passed: bool
    total_score: float
    wer: Optional[float] = None
    cer: Optional[float] = None
    entity_pronunciation_score: Optional[float] = None
    speaker_similarity: Optional[float] = None


class CandidateSelector:
    """Selects the best synthesis candidate using a 2-tier fast/slow evaluation policy."""

    def __init__(self):
        self.artifact_detector = AudioArtifactDetector()

    def evaluate_fast_path(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        planned_duration_sec: float,
    ) -> ArtifactAssessment:
        """Fast validation path (low latency): checks clipping, silence, duration, and repetition."""
        return self.artifact_detector.assess(
            waveform=waveform,
            sample_rate=sample_rate,
            planned_duration_sec=planned_duration_sec,
            asr_target_overlap=1.0,
            phoneme_alignment_score=1.0,
            token_diversity=1.0,
        )

    def select_staged_candidate(
        self,
        synthesis_fn: Callable[[int], Tuple[torch.Tensor, int]],
        planned_duration_sec: float,
        run_slow_path: bool = False,
        slow_evaluator_fn: Optional[Callable[[torch.Tensor, int], Dict[str, float]]] = None,
        max_candidates: int = 3,
    ) -> Tuple[torch.Tensor, int, CandidateEvaluation]:
        """Staged candidate selection pipeline.

        1. Generate Candidate 1.
        2. Run fast validation. If passed and run_slow_path is False -> return immediately.
        3. If failed or run_slow_path is True -> generate Candidate 2 & 3 as needed and pick highest scoring candidate.
        """
        evaluations: List[CandidateEvaluation] = []

        for idx in range(1, max_candidates + 1):
            try:
                waveform, sr = synthesis_fn(idx)
            except Exception as e:
                logger.warning(f"Candidate {idx} synthesis failed: {e}")
                continue

            # Tier 1: Fast validation
            assessment = self.evaluate_fast_path(waveform, sr, planned_duration_sec)
            fast_passed = assessment.is_valid

            wer, cer, entity_score, spk_sim = None, None, None, None
            score = 1.0 - assessment.gibberish_score

            # Tier 2: Slow validation path (only if requested or fast path failed)
            if run_slow_path or not fast_passed:
                if slow_evaluator_fn is not None:
                    try:
                        metrics = slow_evaluator_fn(waveform, sr)
                        wer = metrics.get("wer")
                        cer = metrics.get("cer")
                        entity_score = metrics.get("entity_pronunciation_score", 1.0)
                        spk_sim = metrics.get("speaker_similarity", 1.0)
                        
                        score = (
                            0.30 * (1.0 - min(1.0, cer or 0.0))
                            + 0.30 * (entity_score or 1.0)
                            + 0.20 * (spk_sim or 1.0)
                            + 0.20 * (1.0 - assessment.gibberish_score)
                        )
                    except Exception as se:
                        logger.warning(f"Slow evaluation failed for candidate {idx}: {se}")

            eval_obj = CandidateEvaluation(
                candidate_index=idx,
                waveform=waveform,
                sample_rate=sr,
                assessment=assessment,
                fast_path_passed=fast_passed,
                total_score=score,
                wer=wer,
                cer=cer,
                entity_pronunciation_score=entity_score,
                speaker_similarity=spk_sim,
            )
            evaluations.append(eval_obj)

            # Early return if fast path passed and slow path not strictly required
            if fast_passed and not run_slow_path:
                logger.info(f"Candidate {idx} passed fast validation path -> returning immediately.")
                return waveform, sr, eval_obj

        if not evaluations:
            raise RuntimeError("All candidate generations failed.")

        # Sort candidates by total score descending
        evaluations.sort(key=lambda x: x.total_score, reverse=True)
        best = evaluations[0]
        logger.info(f"Selected candidate {best.candidate_index} (score={best.total_score:.3f}, fast_passed={best.fast_path_passed})")
        return best.waveform, best.sample_rate, best
