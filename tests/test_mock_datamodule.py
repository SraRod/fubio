"""Sprint 0 gate: MockDataModule produces valid BatchDict with landmark dots."""

from __future__ import annotations

import numpy as np
import torch

from fubio.data.task_registry import TASKS
from fubio.train.mock_datamodule import MockDataModule, MockDataset, collate_model


class TestMockDataset:
    def test_deterministic(self) -> None:
        ds1 = MockDataset(n_samples=10, seed=42)
        ds2 = MockDataset(n_samples=10, seed=42)
        s1 = ds1[3]
        s2 = ds2[3]
        img1 = s1["image"]
        img2 = s2["image"]
        assert isinstance(img1, np.ndarray)
        assert isinstance(img2, np.ndarray)
        np.testing.assert_array_equal(img1, img2)

    def test_sample_structure(self) -> None:
        ds = MockDataset(n_samples=5, seed=0)
        sample = ds[0]
        img = sample["image"]
        instances = sample["instances"]

        assert isinstance(img, np.ndarray)
        assert img.shape == (518, 518, 3)
        assert img.dtype == np.uint8

        assert isinstance(instances, list)
        assert 1 <= len(instances) <= 3

    def test_instance_shapes_match_task(self) -> None:
        ds = MockDataset(n_samples=50, seed=0)
        tasks_seen: set[int] = set()
        for i in range(len(ds)):
            sample = ds[i]
            instances = sample["instances"]
            assert isinstance(instances, list)
            for inst in instances:
                assert isinstance(inst, dict)
                task_int = inst["task_id"]
                assert isinstance(task_int, int)
                tasks_seen.add(task_int)
                task_id = next(t.task_id for t in TASKS.values() if t.task_int == task_int)
                k = TASKS[task_id].n_keypoints
                assert inst["keypoints"].shape == (k, 2)
                assert inst["supervised_mask"].shape == (k,)
                assert inst["visible_mask"].shape == (k,)
                assert inst["bbox"].shape == (4,)
                assert inst["transform_matrix"].shape == (3, 3)
                assert inst["original_hw"].shape == (2,)
                assert inst["is_labeled"] is True

        assert len(tasks_seen) >= 5, f"Only saw {len(tasks_seen)} tasks in 50 samples"

    def test_keypoints_in_range(self) -> None:
        ds = MockDataset(n_samples=10, seed=42)
        for i in range(len(ds)):
            instances = ds[i]["instances"]
            assert isinstance(instances, list)
            for inst in instances:
                assert isinstance(inst, dict)
                kp = inst["keypoints"]
                assert isinstance(kp, np.ndarray)
                assert kp.min() >= 0.05, "Keypoints too close to edge"
                assert kp.max() <= 0.95, "Keypoints too close to edge"

    def test_landmark_dots_visible_in_image(self) -> None:
        """Dots should create bright pixels distinguishable from background."""
        ds = MockDataset(n_samples=5, seed=42)
        sample = ds[0]
        img_norm = sample["image"]
        instances = sample["instances"]

        assert isinstance(img_norm, np.ndarray)
        assert isinstance(instances, list)

        # Denormalize to check pixel values
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_01 = img_norm * std + mean

        for inst in instances:
            assert isinstance(inst, dict)
            kp = inst["keypoints"]
            assert isinstance(kp, np.ndarray)
            for j in range(kp.shape[0]):
                x, y = kp[j]
                px = int(x * 518)
                py = int(y * 518)
                px = min(max(px, 0), 517)
                py = min(max(py, 0), 517)
                pixel = img_01[py, px]
                bg_mean = img_01.mean()
                # Dot pixel should differ from background mean
                assert abs(pixel.max() - bg_mean) > 0.05 or pixel.max() > 0.5, (
                    f"Dot at ({px},{py}) not distinguishable from background"
                )


class TestCollateModel:
    def test_batch_dict_structure(self) -> None:
        ds = MockDataset(n_samples=8, seed=0)
        batch_raw = [ds[i] for i in range(4)]
        batch = collate_model(batch_raw)

        assert "image" in batch
        assert "targets" in batch
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape == (4, 3, 518, 518)
        assert batch["image"].dtype == torch.uint8

        targets = batch["targets"]
        assert isinstance(targets, list)
        assert len(targets) == 4
        for sample_targets in targets:
            assert isinstance(sample_targets, list)
            assert len(sample_targets) >= 1

    def test_chw_conversion(self) -> None:
        ds = MockDataset(n_samples=2, seed=0)
        sample = ds[0]
        img_hwc = sample["image"]
        assert isinstance(img_hwc, np.ndarray)

        batch = collate_model([sample])
        img_chw = batch["image"][0]
        # CHW channel 0 should match HWC[:, :, 0]
        np.testing.assert_allclose(
            img_chw[0].numpy(),
            img_hwc[:, :, 0],
            atol=1e-6,
        )


class TestMockDataModule:
    def test_setup_and_iterate(self) -> None:
        dm = MockDataModule(n_train=8, n_val=4, batch_size=2, seed=42)
        dm.setup()

        train_dl = dm.train_dataloader()
        batch = next(iter(train_dl))
        assert batch["image"].shape[0] == 2
        assert len(batch["targets"]) == 2

        val_dl = dm.val_dataloader()
        val_batch = next(iter(val_dl))
        assert val_batch["image"].shape[0] == 2

    def test_train_val_disjoint(self) -> None:
        """Train and val datasets use different seeds → different data."""
        dm = MockDataModule(n_train=4, n_val=4, batch_size=4, seed=42)
        dm.setup()

        train_batch = next(iter(dm.train_dataloader()))
        val_batch = next(iter(dm.val_dataloader()))

        # Images should not be identical
        assert not torch.allclose(train_batch["image"], val_batch["image"])
