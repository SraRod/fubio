"""Unlabeled stream: loader shape, tier flags, task balance, and pool coverage.

Coverage is the load-bearing property here. The unlabeled sampler yields ~65x
more batches per epoch than the labeled one, and CombinedLoader("min_size")
reads only a prefix of them — so whether all ~191K images are ever used, as the
challenge rules require, is decided entirely by the sampler's cursor behaviour
and by nothing that a training run would report.

Upstream: train/datamodule.py, data/sampler.py.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from fubio.data.sampler import MixedTaskBatchSampler
from fubio.train.config import DataConfig, SemiConfig, TransformConfig
from fubio.train.datamodule import FUBioDataModule

MANIFEST = Path("data/manifest.parquet")
requires_data = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not present")


def _data_config() -> DataConfig:
    return DataConfig(
        batch_size=64,
        num_workers=0,
        transform=TransformConfig(target_size=(224, 224)),
    )


# ---------------------------------------------------------------------------
# Sampler cursor — the coverage mechanism, testable without any data
# ---------------------------------------------------------------------------


def _consume(sampler: MixedTaskBatchSampler, n_batches: int) -> list[int]:
    # islice, not enumerate+break: the latter pulls one batch past the limit to
    # test the condition, silently advancing the cursor over indices it discards.
    drawn: list[int] = []
    for batch in islice(sampler, n_batches):
        drawn.extend(batch)
    return drawn


def test_default_sampler_restarts_every_epoch() -> None:
    """Without a persistent cursor, a prefix read keeps re-drawing the same head.

    This is the behaviour that would leave most of a large pool unseen; the test
    pins it so the contrast with persist_cursor stays visible.
    """
    indices = {"A": list(range(100)), "B": list(range(100, 200))}
    sampler = MixedTaskBatchSampler(indices, batch_size=2, seed=0)

    first = set(_consume(sampler, 5))
    sampler.set_epoch(1)
    second = set(_consume(sampler, 5))

    assert len(first | second) < 20, "expected heavy overlap between epochs"


def test_persistent_cursor_covers_the_whole_pool() -> None:
    """Prefix reads across epochs rotate through every index exactly once."""
    indices = {"A": list(range(100)), "B": list(range(100, 200))}
    sampler = MixedTaskBatchSampler(indices, batch_size=2, seed=0, persist_cursor=True)

    drawn: list[int] = []
    for epoch in range(50):  # 50 epochs x 1 index per task per batch x 5 batches
        sampler.set_epoch(epoch)
        drawn.extend(_consume(sampler, 5))

    assert set(drawn) == set(range(200)), "every index must be drawn at least once"
    assert len(drawn) == 500


def test_persistent_cursor_survives_abandoned_iteration() -> None:
    """The cursor must advance even when the consumer stops mid-epoch.

    CombinedLoader("min_size") always stops this sampler early, so state written
    only after the loop would never be written at all.
    """
    indices = {"A": list(range(100))}
    sampler = MixedTaskBatchSampler(indices, batch_size=1, seed=0, persist_cursor=True)

    first = _consume(sampler, 3)
    second = _consume(sampler, 3)

    assert set(first).isdisjoint(second), "second read repeated the first"


# ---------------------------------------------------------------------------
# DataModule wiring
# ---------------------------------------------------------------------------


@requires_data
def test_disabled_returns_a_plain_dataloader() -> None:
    """semi.enabled=False must reproduce the supervised path exactly."""
    dm = FUBioDataModule(_data_config(), semi_config=SemiConfig(enabled=False))
    dm.setup("fit")

    assert not isinstance(dm.train_dataloader(), CombinedLoader)
    assert dm._unlabeled_ds is None, "no unlabeled dataset should be built"


@requires_data
def test_enabled_returns_both_streams() -> None:
    """Both tiers arrive keyed by name, sized independently."""
    dm = FUBioDataModule(_data_config(), semi_config=SemiConfig(enabled=True, batch_size=16))
    dm.setup("fit")
    loader = dm.train_dataloader()

    assert isinstance(loader, CombinedLoader)
    batch = next(iter(loader))[0]
    assert set(batch) == {"labeled", "unlabeled"}
    assert batch["labeled"]["image"].shape[0] == 64
    assert batch["unlabeled"]["image"].shape[0] == 16


@requires_data
def test_unlabeled_instances_are_flagged_unlabeled() -> None:
    """is_labeled=False is what routes these rows to L_mil rather than L_pos."""
    dm = FUBioDataModule(_data_config(), semi_config=SemiConfig(enabled=True, batch_size=16))
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))[0]["unlabeled"]

    for instances in batch["targets"]:
        assert instances, "every unlabeled image must still declare its task"
        for inst in instances:
            assert inst["is_labeled"] is False


@requires_data
def test_epoch_length_is_set_by_the_labeled_stream() -> None:
    """min_size keeps the schedule identical to a supervised run."""
    dm = FUBioDataModule(_data_config(), semi_config=SemiConfig(enabled=True, batch_size=16))
    dm.setup("fit")
    dm.train_dataloader()

    assert dm._unlabeled_sampler is not None
    assert len(dm._unlabeled_sampler) > len(dm._train_sampler)


@requires_data
def test_unlabeled_batches_are_task_balanced() -> None:
    """Every task with an unlabeled pool appears; fetal_femur has none, so cannot."""
    dm = FUBioDataModule(_data_config(), semi_config=SemiConfig(enabled=True, batch_size=16))
    dm.setup("fit")
    dm.train_dataloader()

    assert dm._unlabeled_sampler is not None
    per_task = dm._unlabeled_sampler._per_task
    assert "fetal_femur" not in per_task
    assert len(per_task) == 8
    assert set(per_task.values()) == {2}, "16 over 8 tasks should be a flat 2 each"
