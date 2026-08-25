"""Balanced multi-task batch sampler for heterogeneous task sizes.

Yields batches with fair per-task representation. Small tasks cycle
(reshuffle and restart) within each epoch; the largest task determines
epoch length.

Upstream: ManifestDataset (provides per-task index lists via get_task_indices).
Downstream: DataLoader (used as batch_sampler argument).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class MixedTaskBatchSampler(Sampler[list[int]]):
    """Batch sampler that balances samples across tasks within each batch.

    Per-task allocation: batch_size // n_tasks per task, remainder
    distributed to the largest tasks first. Epoch length is determined
    by the largest task — smaller tasks cycle with reshuffling.

    persist_cursor changes what an "epoch" means for the pool. Default False:
    every epoch reshuffles from scratch, so a consumer that reads only a prefix
    of the batches keeps re-drawing from the front of a fresh permutation and
    never reaches most of a large pool. True: the read position carries across
    epochs and a task's queue reshuffles only once exhausted, which turns partial
    reads into a deterministic rotation that covers the whole pool.

    That matters for the unlabeled loader specifically. Paired with the labeled
    loader under CombinedLoader("min_size"), it yields ~490 of its ~31.7K
    batches per epoch; reshuffling each time would leave ~24% of the largest
    task (A4C, 63K images) never sampled across a 90-epoch run, against a rule
    that requires all ~191K unlabeled images to be used.

    Not checkpoint state: resuming mid-run restarts the cursor, degrading
    coverage toward the reshuffle-every-epoch behaviour rather than breaking.
    """

    def __init__(
        self,
        task_indices: dict[str, list[int]],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
        persist_cursor: bool = False,
    ) -> None:
        if not task_indices:
            raise ValueError("task_indices must be non-empty")
        if batch_size < len(task_indices):
            raise ValueError(f"batch_size ({batch_size}) must be >= n_tasks ({len(task_indices)})")

        self._task_indices = {k: list(v) for k, v in task_indices.items()}
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._seed = seed
        self._epoch = 0
        self._persist_cursor = persist_cursor
        self._queues: dict[str, list[int]] | None = None
        self._positions: dict[str, int] = {}

        # Stable task ordering (deterministic iteration)
        self._task_order = sorted(task_indices.keys())
        n_tasks = len(self._task_order)

        # Per-task allocation: equal split, remainder to largest first
        base = batch_size // n_tasks
        remainder = batch_size % n_tasks
        sizes = sorted(
            self._task_order,
            key=lambda t: len(task_indices[t]),
            reverse=True,
        )
        self._per_task: dict[str, int] = {}
        for i, task_id in enumerate(sizes):
            self._per_task[task_id] = base + (1 if i < remainder else 0)

        # Epoch length: ceil(max_task_size / per_task_allocation)
        max_task = max(self._task_order, key=lambda t: len(task_indices[t]))
        max_size = len(task_indices[max_task])
        alloc = self._per_task[max_task]
        self._n_batches = max_size // alloc
        if not drop_last and max_size % alloc != 0:
            self._n_batches += 1

        logger.info(
            "MixedTaskBatchSampler: %d tasks, batch_size=%d, %d batches/epoch",
            n_tasks,
            batch_size,
            self._n_batches,
        )
        for t in self._task_order:
            logger.info("  %s: %d indices, %d/batch", t, len(task_indices[t]), self._per_task[t])

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for DDP reproducibility."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._epoch)

        if self._persist_cursor:
            if self._queues is None:
                self._queues = self._build_queues(rng)
                self._positions = dict.fromkeys(self._task_order, 0)
            # Aliases, not copies: the loop below mutates the persistent state as
            # it goes. Assigning it back after the loop would not work — a
            # consumer that stops early (CombinedLoader "min_size" reads ~1.5% of
            # this sampler's batches) abandons the generator before its tail runs.
            queues = self._queues
            positions = self._positions
        else:
            queues = self._build_queues(rng)
            positions = dict.fromkeys(self._task_order, 0)

        for _ in range(self._n_batches):
            batch: list[int] = []
            for task_id in self._task_order:
                n_needed = self._per_task[task_id]
                q = queues[task_id]
                pos = positions[task_id]

                for _ in range(n_needed):
                    if pos >= len(q):
                        # Cycle: reshuffle and restart
                        rng_cycle = random.Random(rng.randint(0, 2**31))
                        rng_cycle.shuffle(q)
                        pos = 0
                    batch.append(q[pos])
                    pos += 1

                positions[task_id] = pos

            yield batch

    def _build_queues(self, rng: random.Random) -> dict[str, list[int]]:
        queues: dict[str, list[int]] = {}
        for task_id in self._task_order:
            q = list(self._task_indices[task_id])
            rng.shuffle(q)
            queues[task_id] = q
        return queues


def worker_init_fn(worker_id: int) -> None:
    """Seed worker RNGs for reproducible multi-worker DataLoader."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = info.seed % (2**32)
    np.random.seed(seed)
    random.seed(seed)
