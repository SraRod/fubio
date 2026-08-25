"""Validation must score exactly what inference emits.

These tests exist because the two paths silently diverged: validation scored the
Hungarian/GT-matched instance query while `serving/predict.py` scored the
confidence-argmax one. Every local metric was therefore oracle-assisted, and on
2026-07-25 that inverted the platform ranking of R15 (online 29.42) vs R17
(online 31.84) — local said R17 was the best model of the four, the platform
said it was the worst.

The adversarial case is the important one: a test that only checks agreement
when the two policies coincide cannot catch the regression it exists to catch.
"""

from __future__ import annotations

import torch

from fubio.evaluation.postprocessing import select_serving_query


class TestServingQuerySelection:
    def test_picks_highest_confidence(self):
        conf = torch.tensor([[0.1, 0.9, 0.3, 0.2]])
        assert select_serving_query(conf).tolist() == [1]

    def test_logits_and_probabilities_agree(self):
        """Sigmoid is monotonic — predict.py passes probabilities, module.py logits."""
        logits = torch.tensor([[-2.0, 1.5, 0.3, -0.7], [3.0, -1.0, 2.9, 0.0]])
        assert torch.equal(
            select_serving_query(logits),
            select_serving_query(logits.sigmoid()),
        )

    def test_accepts_trailing_singleton_dim(self):
        """TaskOutput.conf is (B, N_inst, 1); callers should not have to squeeze."""
        conf = torch.tensor([[0.1, 0.9, 0.3, 0.2]])
        assert torch.equal(
            select_serving_query(conf),
            select_serving_query(conf.unsqueeze(-1)),
        )

    def test_ignores_ground_truth_proximity(self):
        """ADVERSARIAL: the query closest to GT is NOT the one with top confidence.

        This is the exact configuration that made the old validation path
        optimistic. Selection must follow confidence and therefore pick the
        WRONG-but-confident query — if it picks the accurate one, something is
        consulting the ground truth.
        """
        gt = torch.tensor([[0.50, 0.50]])
        landmarks = torch.tensor(
            [
                [[0.51, 0.49]],  # query 0 — nearly exact, low confidence
                [[0.90, 0.10]],  # query 1 — far off, high confidence
            ]
        )
        conf = torch.tensor([[0.2, 0.8]])

        chosen = int(select_serving_query(conf).item())
        assert chosen == 1, "selection must follow confidence, not GT distance"

        err = torch.linalg.vector_norm(landmarks - gt, dim=-1).squeeze(-1)
        assert err.argmin().item() == 0, "test is vacuous unless query 0 is closer"
        assert err[chosen] > err[0], "served query must be the worse one here"


class TestBothCallersUseSharedPolicy:
    def test_module_and_predict_import_the_same_function(self):
        """Guards against either path growing its own private copy of the policy."""
        import fubio.serving.predict as predict_mod
        import fubio.train.module as module_mod

        assert module_mod.select_serving_query is select_serving_query
        assert predict_mod.select_serving_query is select_serving_query
