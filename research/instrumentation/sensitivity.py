"""Matrix-free local sensitivity tools.

The functions are model-agnostic so their numerical correctness can be tested on
CPU.  GPU experiments must supply the genuine frozen F5 vector field and acoustic
feature functions; this module never substitutes random tensors for model output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


TensorFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class DerivativeCheck:
    absolute_error: float
    relative_error: float
    autodiff_norm: float
    finite_difference_norm: float


@dataclass(frozen=True)
class SensitivityEstimate:
    target_gain: float
    protected_gain: float
    selectivity_ratio: float
    control_energy: float


def matrix_free_jvp(function: TensorFunction, point: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
    """Return `function(point)` and J(point) @ direction without materializing J."""
    if point.shape != direction.shape:
        raise ValueError("point and direction must have identical shapes")
    return torch.func.jvp(function, (point,), (direction,))


def matrix_free_vjp(function: TensorFunction, point: Tensor, cotangent: Tensor) -> tuple[Tensor, Tensor]:
    """Return `function(point)` and J(point).T @ cotangent."""
    output, pullback = torch.func.vjp(function, point)
    if output.shape != cotangent.shape:
        raise ValueError("cotangent must have the same shape as function output")
    return output, pullback(cotangent)[0]


def central_finite_difference(
    function: TensorFunction, point: Tensor, direction: Tensor, epsilon: float = 1e-4
) -> Tensor:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)


def verify_jvp(
    function: TensorFunction, point: Tensor, direction: Tensor, epsilon: float = 1e-4
) -> DerivativeCheck:
    _, automatic = matrix_free_jvp(function, point, direction)
    numerical = central_finite_difference(function, point, direction, epsilon)
    absolute = torch.linalg.vector_norm(automatic - numerical)
    denominator = torch.maximum(
        torch.linalg.vector_norm(automatic),
        torch.tensor(torch.finfo(automatic.dtype).eps, device=automatic.device, dtype=automatic.dtype),
    )
    return DerivativeCheck(
        absolute_error=float(absolute.detach().cpu()),
        relative_error=float((absolute / denominator).detach().cpu()),
        autodiff_norm=float(torch.linalg.vector_norm(automatic).detach().cpu()),
        finite_difference_norm=float(torch.linalg.vector_norm(numerical).detach().cpu()),
    )


def target_protected_sensitivity(
    output_jvp: Tensor,
    target_mask: Tensor,
    protected_mask: Tensor,
    direction: Tensor,
    ridge: float = 1e-12,
) -> SensitivityEstimate:
    """Measure RMS target/protected response per supplied local direction.

    Masks must be disjoint and broadcastable to the acoustic output. Alignment
    uncertainty should be excluded from both masks by the caller.
    """
    target_mask = target_mask.to(dtype=torch.bool, device=output_jvp.device).expand_as(output_jvp)
    protected_mask = protected_mask.to(dtype=torch.bool, device=output_jvp.device).expand_as(output_jvp)
    if torch.any(target_mask & protected_mask):
        raise ValueError("target and protected masks must be disjoint")
    if not torch.any(target_mask) or not torch.any(protected_mask):
        raise ValueError("both masks must select at least one output element")
    target_gain = torch.sqrt(torch.mean(output_jvp[target_mask].square()))
    protected_gain = torch.sqrt(torch.mean(output_jvp[protected_mask].square()))
    energy = torch.sum(direction.square())
    ratio = target_gain / (protected_gain + ridge)
    return SensitivityEstimate(
        target_gain=float(target_gain.detach().cpu()),
        protected_gain=float(protected_gain.detach().cpu()),
        selectivity_ratio=float(ratio.detach().cpu()),
        control_energy=float(energy.detach().cpu()),
    )


def generalized_selective_basis(
    target_responses: Tensor, protected_responses: Tensor, ridge: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Solve the finite-basis target/protected generalized Rayleigh problem.

    Columns correspond to candidate control basis vectors; rows are flattened
    acoustic features. Returned eigenvalues are descending and eigenvectors are
    coefficients in that control basis. This is local linear algebra, not a
    nonlinear preservation guarantee.
    """
    if target_responses.ndim != 2 or protected_responses.ndim != 2:
        raise ValueError("response matrices must be rank two")
    if target_responses.shape[1] != protected_responses.shape[1]:
        raise ValueError("response matrices must share control-basis dimension")
    target_gram = target_responses.T @ target_responses
    protected_gram = protected_responses.T @ protected_responses
    eye = torch.eye(protected_gram.shape[0], device=protected_gram.device, dtype=protected_gram.dtype)
    denominator = protected_gram + ridge * eye
    chol = torch.linalg.cholesky(denominator)
    whitened = torch.linalg.solve_triangular(chol, target_gram, upper=False)
    whitened = torch.linalg.solve_triangular(chol, whitened.T, upper=False).T
    whitened = (whitened + whitened.T) / 2
    values, vectors_white = torch.linalg.eigh(whitened)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors_white = vectors_white[:, order]
    vectors = torch.linalg.solve_triangular(chol.T, vectors_white, upper=True)
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=0, keepdim=True).clamp_min(ridge)
    return values, vectors

