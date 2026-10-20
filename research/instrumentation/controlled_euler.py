"""Differentiable controlled Euler integration for frozen vector fields."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


VectorField = Callable[[Tensor, Tensor], Tensor]
ControlMap = Callable[[int, Tensor, Tensor], Tensor]


def controlled_euler(
    vector_field: VectorField,
    initial_state: Tensor,
    times: Tensor,
    control: ControlMap | None = None,
) -> tuple[Tensor, Tensor]:
    """Integrate `dx/dt=v(x,t)+control(step,x,t)` on the supplied time grid.

    The routine keeps the graph intact. Genuine experiments must construct
    `vector_field` from the verified frozen F5 DiT including its actual CFG and
    conditioning path.
    """
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be a one-dimensional grid of length >= 2")
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing")
    state = initial_state
    trajectory = [state]
    for index in range(times.numel() - 1):
        time = times[index]
        velocity = vector_field(state, time)
        if velocity.shape != state.shape:
            raise ValueError("vector field output must match state shape")
        if control is not None:
            intervention = control(index, state, time)
            if intervention.shape != state.shape:
                raise ValueError("control output must match state shape")
            velocity = velocity + intervention
        state = state + (times[index + 1] - time) * velocity
        trajectory.append(state)
    return state, torch.stack(trajectory)

