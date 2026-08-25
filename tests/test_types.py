"""Sprint 0 gate: data/types.py contracts compile and shapes are correct."""

from __future__ import annotations

import numpy as np
import torch

from fubio.data.task_registry import TASKS
from fubio.data.types import BatchDict, InstanceDict, TaskOutput


def _make_instance(task_int: int = 0, k: int = 16) -> InstanceDict:
    return {
        "task_id": task_int,
        "keypoints": np.random.rand(k, 2).astype(np.float32),
        "supervised_mask": np.ones(k, dtype=bool),
        "visible_mask": np.ones(k, dtype=bool),
        "bbox": np.array([0.5, 0.5, 0.3, 0.3], dtype=np.float32),
        "transform_matrix": np.eye(3, dtype=np.float32),
        "original_hw": np.array([518, 518], dtype=np.int32),
        "is_labeled": True,
        "image_path": "test.png",
    }


class TestInstanceDict:
    def test_all_tasks_constructible(self) -> None:
        for task in TASKS.values():
            inst = _make_instance(task.task_int, task.n_keypoints)
            assert inst["keypoints"].shape == (task.n_keypoints, 2)
            assert inst["supervised_mask"].shape == (task.n_keypoints,)
            assert inst["visible_mask"].shape == (task.n_keypoints,)

    def test_bbox_shape(self) -> None:
        inst = _make_instance()
        assert inst["bbox"].shape == (4,)
        assert inst["bbox"].dtype == np.float32

    def test_transform_matrix_shape(self) -> None:
        inst = _make_instance()
        assert inst["transform_matrix"].shape == (3, 3)


class TestBatchDict:
    def test_construction(self) -> None:
        batch: BatchDict = {
            "image": torch.randn(2, 3, 518, 518),
            "targets": [[_make_instance()], [_make_instance(1, 22)]],
        }
        assert batch["image"].shape == (2, 3, 518, 518)
        assert len(batch["targets"]) == 2
        assert batch["targets"][0][0]["task_id"] == 0
        assert batch["targets"][1][0]["task_id"] == 1


class TestTaskOutput:
    def test_named_fields(self) -> None:
        out = TaskOutput(
            bbox=torch.rand(2, 2, 4),
            conf=torch.randn(2, 2, 1),
            landmarks=torch.rand(2, 2, 16, 2),
        )
        assert out.bbox.shape == (2, 2, 4)
        assert out.conf.shape == (2, 2, 1)
        assert out.landmarks.shape == (2, 2, 16, 2)

    def test_unpacking(self) -> None:
        out = TaskOutput(
            bbox=torch.rand(1, 2, 4),
            conf=torch.randn(1, 2, 1),
            landmarks=torch.rand(1, 2, 4, 2),
        )
        assert out.bbox.shape == (1, 2, 4)
        assert out.conf.shape == (1, 2, 1)
        assert out.landmarks.shape == (1, 2, 4, 2)
        assert out.residual is None

    def test_per_task_shapes(self) -> None:
        for tid, tdef in TASKS.items():
            out = TaskOutput(
                bbox=torch.rand(2, 2, 4),
                conf=torch.randn(2, 2, 1),
                landmarks=torch.rand(2, 2, tdef.n_keypoints, 2),
            )
            assert out.landmarks.shape == (2, 2, tdef.n_keypoints, 2), tid
