"""Random utilities with optional MATLAB-compatible behavior."""

from __future__ import annotations

import numpy as np

RNG = np.random.Generator | np.random.RandomState


def make_rng(seed: int | None, rng_type: str) -> RNG:
    """Create an RNG matching numpy or MATLAB's MT19937."""
    if rng_type == "matlab":
        # MATLAB's default generator is MT19937 ("twister")
        return np.random.RandomState(seed)
    return np.random.default_rng(seed)


def uniform(rng: RNG, size: int | tuple[int, ...]) -> np.ndarray:
    """Draw uniform(0,1) samples using Generator or RandomState."""
    if isinstance(rng, np.random.Generator):
        return rng.random(size)
    return rng.random_sample(size)


def sample_with_replacement(
    population: np.ndarray | list[int],
    size: int,
    rng: RNG,
    matlab_compat: bool = False,
) -> np.ndarray:
    """Sample with replacement from population.

    When matlab_compat=True, use a uniform-to-index mapping akin to randi/datasample.
    """
    pop = np.asarray(population)
    if pop.size == 0:
        raise ValueError("Population is empty.")
    if not matlab_compat:
        if isinstance(rng, np.random.Generator):
            idx = rng.choice(pop.size, size=size, replace=True)
        else:
            idx = rng.choice(pop.size, size=size, replace=True)
        return pop[idx]

    u = uniform(rng, size)
    idx = (u * pop.size).astype(int)
    return pop[idx]
