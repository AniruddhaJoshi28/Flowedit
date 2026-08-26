"""
Candidate Validation Gate for FlowEdit (arXiv:2606.20518).

Safety criterion ensuring learned perturbation δ* meets convergence and stability requirements
before being committed to persistent Modern Hopfield Memory.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    accepted: bool
    stability_passed: bool
    loss_passed: bool
    reason: str


class CandidateValidator:
    """Evaluates candidate FlowEdit corrections before memory insertion."""

    def __init__(
        self,
        max_relative_delta: float = 3.0,
    ):
        self.max_relative_delta = max_relative_delta

    def validate(
        self,
        optimization_result: Any,
        word: str,
        initial_loss: float,
        final_loss: float,
    ) -> ValidationResult:
        """Validate candidate correction."""
        reasons = []

        # 1. Stability check: 0 < ||δ_I|| / ||c_I|| ≤ max_relative_delta
        rel_ratio = getattr(optimization_result, "relative_delta_ratio", 0.0)
        delta_norm = getattr(optimization_result, "delta_norm", 0.0)
        stability_passed = (delta_norm > 0.01) and (rel_ratio <= (self.max_relative_delta + 1e-3))
        if delta_norm <= 0.01:
            reasons.append(f"Zero or negligible delta norm ({delta_norm:.4f})")
        elif rel_ratio > (self.max_relative_delta + 1e-3):
            reasons.append(f"Relative delta ratio {rel_ratio:.4f} > limit {self.max_relative_delta}")

        # 2. Loss convergence check: final_loss must improve upon initial_loss
        loss_passed = (final_loss < initial_loss) and getattr(optimization_result, "converged", True)
        if not loss_passed:
            reasons.append(f"Loss failed to improve ({initial_loss:.4f} → {final_loss:.4f})")

        accepted = stability_passed and loss_passed
        reason = "Passed validation" if accepted else " | ".join(reasons)

        logger.info(f"[Candidate Validation] Target: '{word}' -> Accepted={accepted} ({reason})")

        return ValidationResult(
            accepted=accepted,
            stability_passed=stability_passed,
            loss_passed=loss_passed,
            reason=reason,
        )
