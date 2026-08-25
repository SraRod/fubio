"""Sprint 2 gate: models/ package shape verification with stub backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from fubio.data.task_registry import TASKS
from fubio.data.types import TaskOutput
from fubio.models.backbone import BackboneOutput
from fubio.models.decoder import CrossAttentionDecoder, CrossAttentionLayer, TaskRefinerLayer
from fubio.models.heads import TaskModule
from fubio.models.neck import LinearNeck, sinusoidal_2d_pos_enc

B = 2
D = 256
N_SPATIAL = 1369  # 37 * 37
C_BACKBONE = 768


# ---------------------------------------------------------------------------
# Stub backbone for tests (no torch.hub download)
# ---------------------------------------------------------------------------


class _StubBackbone(nn.Module):
    """Mimics DINOv2Backbone output shapes without loading real weights."""

    def __init__(self, c_backbone: int = C_BACKBONE) -> None:
        super().__init__()
        self.embed_dim = c_backbone
        self.patch_size = 14
        self._linear = nn.Linear(3, c_backbone)

    def freeze(self) -> None:
        pass

    def unfreeze(self) -> None:
        pass

    def param_groups(self, lr: float, layer_decay: float = 1.0) -> list[dict]:
        return [{"params": list(self.parameters()), "lr": lr, "name": "backbone"}]

    def forward(self, x: Tensor) -> BackboneOutput:
        b = x.shape[0]
        tokens = torch.randn(b, N_SPATIAL, self.embed_dim, device=x.device)
        return BackboneOutput(
            features=[tokens],
            spatial_shape=(37, 37),
        )


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
# Projection
# ---------------------------------------------------------------------------


class TestLinearNeck:
    def test_shapes(self) -> None:
        neck = LinearNeck(C_BACKBONE, D)
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
        tm = TaskModule(n_keypoints=K, d_model=D, n_inst=n_inst)
        q, qp = tm.get_queries(batch_size=B)
        assert q.shape == (B, n_inst * (1 + K), D)
        # Without anchors the positional term is zeros, not None — callers add it
        # unconditionally and zeros are the no-op under addition.
        assert qp.shape == q.shape
        assert torch.equal(qp, torch.zeros_like(qp))

    def test_n_queries(self) -> None:
        n_inst = 2
        for tid, tdef in TASKS.items():
            tm = TaskModule(n_keypoints=tdef.n_keypoints, d_model=D, n_inst=n_inst)
            expected = n_inst * (1 + tdef.n_keypoints)
            assert tm.n_queries == expected, f"{tid}: expected {expected}, got {tm.n_queries}"

    def test_anchor_queries(self) -> None:
        K, n_inst = 4, 2
        tm = TaskModule(n_keypoints=K, d_model=D, n_inst=n_inst, use_anchors=True)
        q, qp = tm.get_queries(batch_size=B)
        assert q.shape == (B, n_inst * (1 + K), D)
        assert qp is not None
        assert qp.shape == (B, n_inst * (1 + K), D)


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
        tm = TaskModule(n_keypoints=K, d_model=D, n_inst=n_inst, n_head_layers=1)
        x = torch.randn(B, n_inst * (1 + K), D)
        out = tm.head(x)
        assert isinstance(out, TaskOutput)
        assert out.bbox.shape == (B, n_inst, 4)
        assert out.conf.shape == (B, n_inst, 1)
        assert out.landmarks.shape == (B, n_inst, K, 2)

    def test_bbox_in_range(self) -> None:
        tm = TaskModule(n_keypoints=6, d_model=D, n_inst=2, n_head_layers=1)
        x = torch.randn(B, 2 * (1 + 6), D)
        out = tm.head(x)
        assert out.bbox.min() >= 0.0
        assert out.bbox.max() <= 1.0

    def test_landmarks_in_range(self) -> None:
        tm = TaskModule(n_keypoints=6, d_model=D, n_inst=2, n_head_layers=1)
        x = torch.randn(B, 2 * (1 + 6), D)
        out = tm.head(x)
        assert out.landmarks.min() >= 0.0
        assert out.landmarks.max() <= 1.0

    def test_multi_head_layers(self) -> None:
        tm = TaskModule(n_keypoints=4, d_model=D, n_inst=2, n_head_layers=3)
        x = torch.randn(B, 2 * (1 + 4), D)
        out = tm.head(x)
        assert out.bbox.shape == (B, 2, 4)

    def test_all_task_shapes(self) -> None:
        """All 9 task modules produce correct output shapes."""
        n_inst = 2
        for tid, tdef in TASKS.items():
            tm = TaskModule(n_keypoints=tdef.n_keypoints, d_model=D, n_inst=n_inst)
            x = torch.randn(B, n_inst * (1 + tdef.n_keypoints), D)
            out = tm.head(x)
            assert out.landmarks.shape == (B, n_inst, tdef.n_keypoints, 2), tid


# ---------------------------------------------------------------------------
# FUBioModel end-to-end (with stub backbone)
# ---------------------------------------------------------------------------


def _make_model() -> nn.Module:
    from fubio.models.model import FUBioModel

    model = FUBioModel.__new__(FUBioModel)
    nn.Module.__init__(model)
    # __init__ is bypassed above, so plain attributes it would set must be
    # mirrored by hand. _geo=False selects the non-GeoSimCC path, which needs
    # neither loc_k_proj nor the fine-detail stem. Keep in sync with
    # FUBioModel.__init__ when it gains a new non-Module attribute.
    model._geo = False
    n_inst = 2
    model.n_inst = n_inst
    model.backbone = _StubBackbone()
    model.neck = LinearNeck(C_BACKBONE, D)
    model.decoder = CrossAttentionDecoder(d_model=D, n_heads=8, ffn_dim=1024, n_layers=2)
    model.tasks = nn.ModuleDict(
        {
            tid: TaskModule(
                n_keypoints=tdef.n_keypoints,
                d_model=D,
                n_inst=n_inst,
            )
            for tid, tdef in TASKS.items()
        }
    )
    return model


class TestFUBioModelForward:
    def test_output_all_tasks(self) -> None:
        model = _make_model()
        images = torch.randn(B, 3, 518, 518)
        model_out = model(images)
        task_outputs = model_out.task_outputs

        assert len(task_outputs) == 9
        for tid, out in task_outputs.items():
            K = TASKS[tid].n_keypoints
            assert isinstance(out, TaskOutput)
            assert out.bbox.shape == (B, 2, 4)
            assert out.conf.shape == (B, 2, 1)
            assert out.landmarks.shape == (B, 2, K, 2)

    def test_bbox_normalized(self) -> None:
        model = _make_model()
        images = torch.randn(B, 3, 518, 518)
        model_out = model(images)
        for out in model_out.task_outputs.values():
            assert out.bbox.min() >= 0.0
            assert out.bbox.max() <= 1.0

    def test_gradient_flows(self) -> None:
        """Gradient flows from landmark output back through decoder."""
        model = _make_model()
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
