import torch

from research.instrumentation.controlled_euler import controlled_euler
from research.instrumentation.sensitivity import (
    generalized_selective_basis,
    matrix_free_jvp,
    matrix_free_vjp,
    target_protected_sensitivity,
    verify_jvp,
)


def test_jvp_matches_analytic_and_central_difference():
    torch.manual_seed(7)
    matrix = torch.randn(5, 4, dtype=torch.float64)
    point = torch.randn(4, dtype=torch.float64)
    direction = torch.randn(4, dtype=torch.float64)
    function = lambda value: torch.sin(matrix @ value)
    output, jvp = matrix_free_jvp(function, point, direction)
    expected = torch.cos(matrix @ point) * (matrix @ direction)
    assert torch.allclose(output, function(point), atol=1e-12, rtol=1e-12)
    assert torch.allclose(jvp, expected, atol=1e-12, rtol=1e-12)
    check = verify_jvp(function, point, direction, epsilon=1e-5)
    assert check.relative_error < 1e-9


def test_vjp_obeys_adjoint_identity():
    torch.manual_seed(11)
    matrix = torch.randn(6, 3, dtype=torch.float64)
    point = torch.randn(3, dtype=torch.float64)
    direction = torch.randn(3, dtype=torch.float64)
    cotangent = torch.randn(6, dtype=torch.float64)
    function = lambda value: torch.tanh(matrix @ value)
    _, jvp = matrix_free_jvp(function, point, direction)
    _, vjp = matrix_free_vjp(function, point, cotangent)
    assert torch.allclose(torch.dot(cotangent, jvp), torch.dot(vjp, direction), atol=1e-11, rtol=1e-11)


def test_target_protected_masks_are_disjoint_and_quantified():
    response = torch.tensor([[3.0, 4.0, 0.5, 0.5]], dtype=torch.float64)
    target = torch.tensor([[True, True, False, False]])
    protected = ~target
    estimate = target_protected_sensitivity(response, target, protected, torch.tensor([2.0]))
    assert estimate.target_gain > estimate.protected_gain
    assert estimate.selectivity_ratio > 7.0
    assert estimate.control_energy == 4.0


def test_generalized_basis_prefers_target_selective_direction():
    target = torch.tensor([[4.0, 0.1], [0.0, 0.1]], dtype=torch.float64)
    protected = torch.tensor([[0.1, 2.0], [0.1, 2.0]], dtype=torch.float64)
    values, vectors = generalized_selective_basis(target, protected, ridge=1e-4)
    assert values[0] > values[1]
    assert abs(vectors[0, 0]) > abs(vectors[1, 0])


def test_controlled_euler_propagates_real_gradient_to_control():
    control_amplitude = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([1.0], dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)

    def field(state, time):
        return -state

    def control(step, state, time):
        return torch.ones_like(state) * control_amplitude

    final, trajectory = controlled_euler(field, initial, times, control)
    final.square().backward()
    assert trajectory.shape == (11, 1)
    assert control_amplitude.grad is not None
    assert torch.isfinite(control_amplitude.grad)
    assert abs(float(control_amplitude.grad)) > 0

