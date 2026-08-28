"""Differential tests: supportive_torch vs the numpy reference implementation.

For every task with supportive landmarks, random plausible scored landmarks are
fed to both `compute_supportive` (numpy, per-sample) and
`compute_supportive_torch` (torch, batched); the outputs must agree within
float32 tolerance. Also verifies gradient flow through the torch path and the
None-returning tasks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fubio.data.supportive import compute_supportive
from fubio.data.supportive_torch import compute_supportive_torch
from fubio.data.task_registry import TASKS

SUPPORTIVE_TASKS = sorted(t.task_id for t in TASKS.values() if t.n_supportive > 0)
NO_SUPPORTIVE_TASKS = sorted(t.task_id for t in TASKS.values() if t.n_supportive == 0)


def _random_scored(task_id: str, n: int, seed: int, scale: float = 640.0) -> np.ndarray:
    """Random plausible scored landmarks in pixel space, (n, K_scored, 2) float32.

    Pixel scale keeps the torch implementation's numerical guards (1e-3
    parallel threshold, 1e-4 eps under sqrt) negligible relative to the
    coordinate magnitude, matching how the numpy reference is used.
    """
    rng = np.random.default_rng(seed)
    k = TASKS[task_id].n_keypoints
    return rng.uniform(0.1 * scale, 0.9 * scale, size=(n, k, 2)).astype(np.float32)


# =========================================================================
# Differential agreement — numpy reference vs torch
# =========================================================================


class TestDifferentialAgreement:
    @pytest.mark.parametrize("task_id", SUPPORTIVE_TASKS)
    def test_matches_numpy_reference(self, task_id: str) -> None:
        """Batched torch output equals per-sample numpy output within tolerance."""
        gt = _random_scored(task_id, n=8, seed=1234)

        np_out = np.stack([compute_supportive(task_id, gt[i])[0] for i in range(len(gt))])

        torch_out = compute_supportive_torch(task_id, torch.from_numpy(gt))
        assert torch_out is not None

        n_sup = TASKS[task_id].n_supportive
        assert torch_out.shape == (len(gt), n_sup, 2)
        assert np_out.shape == (len(gt), n_sup, 2)
        np.testing.assert_allclose(
            torch_out.numpy(), np_out, rtol=1e-4, atol=1e-2,
            err_msg=f"torch/numpy supportive mismatch for {task_id}",
        )

    @pytest.mark.parametrize("task_id", SUPPORTIVE_TASKS)
    def test_matches_numpy_in_unit_space(self, task_id: str) -> None:
        """Agreement also holds in [0, 1] coordinates (the torch input contract)."""
        gt = (_random_scored(task_id, n=4, seed=99) / 640.0).astype(np.float32)

        np_out = np.stack([compute_supportive(task_id, gt[i])[0] for i in range(len(gt))])
        torch_out = compute_supportive_torch(task_id, torch.from_numpy(gt))
        assert torch_out is not None
        # atol dominated by the torch ellipse eps (sqrt(|a²-b²| + 1e-4)):
        # worst case shifts a focus by ~1e-2 in unit space when a ≈ b.
        np.testing.assert_allclose(torch_out.numpy(), np_out, rtol=1e-3, atol=2e-2)

    def test_a4c_parallel_lines_fall_back_to_centroid(self) -> None:
        """Exactly parallel chamber axes: both implementations return the centroid."""
        gt = _random_scored("A4C", n=1, seed=7)
        # Chamber 0 uses landmarks 0..3: make line(0→1) parallel to line(2→3).
        gt[0, 0] = [100.0, 100.0]
        gt[0, 1] = [200.0, 200.0]
        gt[0, 2] = [100.0, 150.0]
        gt[0, 3] = [200.0, 250.0]

        np_out = compute_supportive("A4C", gt[0])[0]
        torch_out = compute_supportive_torch("A4C", torch.from_numpy(gt))
        assert torch_out is not None

        centroid = gt[0, :4].mean(axis=0)
        np.testing.assert_allclose(np_out[0], centroid, rtol=1e-5)
        np.testing.assert_allclose(torch_out[0, 0].numpy(), centroid, rtol=1e-5)
        np.testing.assert_allclose(torch_out[0].numpy(), np_out, rtol=1e-4, atol=1e-2)


# =========================================================================
# Gradient flow
# =========================================================================


class TestGradientFlow:
    @pytest.mark.parametrize("task_id", SUPPORTIVE_TASKS)
    def test_backward_produces_finite_grads(self, task_id: str) -> None:
        gt = torch.from_numpy(_random_scored(task_id, n=3, seed=5) / 640.0)
        gt.requires_grad_(True)

        out = compute_supportive_torch(task_id, gt)
        assert out is not None
        out.sum().backward()

        assert gt.grad is not None
        assert torch.isfinite(gt.grad).all()
        # Every supportive point depends on scored landmarks — grad must be nonzero.
        assert gt.grad.abs().sum().item() > 0.0

    def test_degenerate_ivc_stays_finite(self) -> None:
        """Coincident IVC endpoints: torch is designed to stay finite (numpy NaNs).

        This is an intentional divergence — the torch path avoids norm/division
        so gradients survive degenerate predictions during training.
        """
        gt = torch.full((1, 2, 2), 0.5, requires_grad=True)
        out = compute_supportive_torch("IVC", gt)
        assert out is not None
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert gt.grad is not None
        assert torch.isfinite(gt.grad).all()

        np_out = compute_supportive("IVC", np.full((2, 2), 0.5, dtype=np.float32))[0]
        assert np.isnan(np_out).all()


# =========================================================================
# Tasks without supportive landmarks
# =========================================================================


class TestNoSupportiveTasks:
    @pytest.mark.parametrize("task_id", NO_SUPPORTIVE_TASKS)
    def test_returns_none(self, task_id: str) -> None:
        gt = torch.from_numpy(_random_scored(task_id, n=2, seed=0))
        assert compute_supportive_torch(task_id, gt) is None

    def test_unknown_task_returns_none(self) -> None:
        gt = torch.zeros(1, 4, 2)
        assert compute_supportive_torch("no_such_task", gt) is None

    @pytest.mark.parametrize("task_id", NO_SUPPORTIVE_TASKS)
    def test_numpy_reference_returns_empty(self, task_id: str) -> None:
        """The numpy counterpart returns an empty result, not None — both mean
        'no supportive'; downstream code must handle each convention."""
        coords, sources = compute_supportive(task_id, _random_scored(task_id, 1, 0)[0])
        assert coords.shape == (0, 2)
        assert sources == ()
