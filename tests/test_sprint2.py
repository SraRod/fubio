"""Sprint 2 gate: models/ package shape verification with stub backbone."""

from __future__ import annotations

import torch

from conftest import C_BACKBONE, N_SPATIAL, D, make_task_module
from fubio.data.task_registry import TASKS
from fubio.data.types import TaskOutput
from fubio.models.backbone import BackboneOutput
from fubio.models.decoder import CrossAttentionDecoder, CrossAttentionLayer, TaskRefinerLayer
from fubio.models.neck import C2fNeck, sinusoidal_2d_pos_enc

B = 2


def _head_inputs(batch: int = B) -> dict:
    """Memory-side inputs TaskModule.head needs for the GeoSimCC readout."""
    return {
        "memory": torch.randn(batch, N_SPATIAL, D),
        "spatial_shape": (37, 37),
        "memory_pos": torch.zeros(1, N_SPATIAL, D),
    }


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class TestSinusoidal2DPosEnc:
    def test_shape(self) -> None:
        pe = sinusoidal_2d_pos_enc(37, 37, D)
        assert pe.shape == (1, 1369, D)

    def test_different_sizes(self) -> None:
        pe = sinusoidal_2d_pos_enc(10, 20, 128)
        assert pe.shape == (1, 200, 128)

    def test_values_bounded(self) -> None:
        pe = sinusoidal_2d_pos_enc(37, 37, D)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0


# ---------------------------------------------------------------------------
# Neck
# ---------------------------------------------------------------------------


class TestC2fNeck:
    def test_shapes(self) -> None:
        neck = C2fNeck(n_layers=1, c_backbone=C_BACKBONE, d_model=D, n_bottleneck=1)
        backbone_out = BackboneOutput(
            features=[torch.randn(B, N_SPATIAL, C_BACKBONE)],
            spatial_shape=(37, 37),
        )
        neck_out = neck(backbone_out)
        memory, memory_pos = neck_out.memory, neck_out.memory_pos
        assert memory.shape == (B, N_SPATIAL, D)
        assert memory_pos.shape == (1, N_SPATIAL, D)


# ---------------------------------------------------------------------------
# TaskModule queries
# ---------------------------------------------------------------------------


class TestTaskModuleQueries:
    def test_shapes(self) -> None:
        K, n_inst = 4, 2
        tm = make_task_module(K, n_inst=n_inst)
        q, qp = tm.get_queries(batch_size=B)
        assert q.shape == (B, n_inst * (1 + K), D)
        # Anchors are initialized from the prior mean, so the positional term
        # is a real encoding, not zeros.
        assert qp.shape == q.shape

    def test_n_queries(self) -> None:
        n_inst = 2
        for tid, tdef in TASKS.items():
            tm = make_task_module(tdef.n_keypoints, n_inst=n_inst)
            expected = n_inst * (1 + tdef.n_keypoints)
            assert tm.n_queries == expected, f"{tid}: expected {expected}, got {tm.n_queries}"


# ---------------------------------------------------------------------------
# Cross-attention layers
# ---------------------------------------------------------------------------


class TestCrossAttentionLayer:
    def test_output_shape(self) -> None:
        layer = CrossAttentionLayer(D, 8, 1024)
        N_q = 50
        queries = torch.randn(B, N_q, D)
        memory = torch.randn(B, N_SPATIAL, D)
        memory_pos = torch.randn(1, N_SPATIAL, D)
        out = layer(queries, memory, memory_pos)
        assert out.shape == (B, N_q, D)


class TestCrossAttentionDecoder:
    def test_output_shape(self) -> None:
        decoder = CrossAttentionDecoder(D, 8, 1024, n_layers=2)
        N_q = 50
        queries = torch.randn(B, N_q, D)
        memory = torch.randn(B, N_SPATIAL, D)
        memory_pos = torch.randn(1, N_SPATIAL, D)
        out = decoder(queries, memory, memory_pos)
        assert out.shape == (B, N_q, D)


class TestTaskRefinerLayer:
    def test_output_shape(self) -> None:
        layer = TaskRefinerLayer(D, 8, 1024)
        x = torch.randn(B, 10, D)
        out = layer(x)
        assert out.shape == (B, 10, D)


# ---------------------------------------------------------------------------
# Cross-attention independence
# ---------------------------------------------------------------------------


class TestCrossAttentionIndependence:
    def test_perturb_one_task_no_effect_on_other(self) -> None:
        """Perturbing one task's queries must not affect another's output."""
        decoder = CrossAttentionDecoder(D, 8, 1024, n_layers=2)
        decoder.eval()

        memory = torch.randn(1, N_SPATIAL, D)
        memory_pos = torch.randn(1, N_SPATIAL, D)
        q_a = torch.randn(1, 10, D)
        q_b = torch.randn(1, 20, D)

        # Run concatenated
        all_q = torch.cat([q_a, q_b], dim=1)
        with torch.no_grad():
            out_all = decoder(all_q, memory, memory_pos)
        out_b_from_all = out_all[:, 10:, :]

        # Perturb q_a, run again
        q_a_perturbed = q_a + torch.randn_like(q_a) * 10.0
        all_q2 = torch.cat([q_a_perturbed, q_b], dim=1)
        with torch.no_grad():
            out_all2 = decoder(all_q2, memory, memory_pos)
        out_b_from_all2 = out_all2[:, 10:, :]

        torch.testing.assert_close(out_b_from_all, out_b_from_all2)


# ---------------------------------------------------------------------------
# TaskModule head
# ---------------------------------------------------------------------------


class TestTaskModuleHead:
    def test_shapes(self) -> None:
        K = 4
        n_inst = 2
        tm = make_task_module(K, n_inst=n_inst)
        x = torch.randn(B, n_inst * (1 + K), D)
        out = tm.head(x, **_head_inputs())
        assert isinstance(out, TaskOutput)
        assert out.bbox.shape == (B, n_inst, 4)
        assert out.conf.shape == (B, n_inst, 1)
        assert out.landmarks.shape == (B, n_inst, K, 2)

    def test_bbox_in_range(self) -> None:
        tm = make_task_module(6, n_inst=2)
        x = torch.randn(B, 2 * (1 + 6), D)
        out = tm.head(x, **_head_inputs())
        assert out.bbox.min() >= 0.0
        assert out.bbox.max() <= 1.0

    def test_landmarks_in_range(self) -> None:
        tm = make_task_module(6, n_inst=2)
        x = torch.randn(B, 2 * (1 + 6), D)
        out = tm.head(x, **_head_inputs())
        assert out.landmarks.min() >= 0.0
        assert out.landmarks.max() <= 1.0

    def test_stage1_heat_is_returned(self) -> None:
        """GeoSimCC routes its stage-1 heat to TaskOutput.heatmap for lambda_heatmap."""
        K, n_inst = 4, 2
        tm = make_task_module(K, n_inst=n_inst)
        x = torch.randn(B, n_inst * (1 + K), D)
        out = tm.head(x, **_head_inputs())
        assert out.heatmap is not None
        assert out.heatmap.shape == (B, n_inst, K, N_SPATIAL)
        # Softmax over the grid: each landmark's heat sums to 1
        torch.testing.assert_close(
            out.heatmap.sum(-1), torch.ones(B, n_inst, K), atol=1e-4, rtol=0
        )

    def test_multi_head_layers(self) -> None:
        tm = make_task_module(4, n_inst=2, n_head_layers=3)
        x = torch.randn(B, 2 * (1 + 4), D)
        out = tm.head(x, **_head_inputs())
        assert out.bbox.shape == (B, 2, 4)

    def test_all_task_shapes(self) -> None:
        """All 9 task modules produce correct output shapes."""
        n_inst = 2
        for tid, tdef in TASKS.items():
            tm = make_task_module(tdef.n_keypoints, n_inst=n_inst)
            x = torch.randn(B, n_inst * (1 + tdef.n_keypoints), D)
            out = tm.head(x, **_head_inputs())
            assert out.landmarks.shape == (B, n_inst, tdef.n_keypoints, 2), tid


# ---------------------------------------------------------------------------
# FUBioModel end-to-end (real constructor, stub backbone)
# ---------------------------------------------------------------------------


class TestFUBioModelForward:
    def test_output_all_tasks(self, stub_backbone) -> None:
        from conftest import make_module

        model = make_module().model
        images = torch.randn(B, 3, 518, 518)
        model_out = model(images)
        task_outputs = model_out.task_outputs

        assert len(task_outputs) == 9
        for tid, out in task_outputs.items():
            K = TASKS[tid].n_keypoints
            assert isinstance(out, TaskOutput)
            assert out.bbox.shape == (B, 1, 4)
            assert out.conf.shape == (B, 1, 1)
            assert out.landmarks.shape == (B, 1, K, 2)

    def test_bbox_normalized(self, stub_backbone) -> None:
        from conftest import make_module

        model = make_module().model
        images = torch.randn(B, 3, 518, 518)
        model_out = model(images)
        for out in model_out.task_outputs.values():
            assert out.bbox.min() >= 0.0
            assert out.bbox.max() <= 1.0

    def test_gradient_flows(self, stub_backbone) -> None:
        """Gradient flows from landmark output back through decoder."""
        from conftest import make_module

        model = make_module().model
        images = torch.randn(B, 3, 518, 518)
        model_out = model(images)
        loss = sum(out.landmarks.sum() for out in model_out.task_outputs.values())
        loss.backward()

        # Decoder parameters should have gradients
        for name, p in model.decoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for decoder.{name}"

        # Task query embeddings should have gradients
        for tid, tm in model.tasks.items():
            assert tm.query_embedding.weight.grad is not None, (
                f"No gradient for {tid} query_embedding"
            )
