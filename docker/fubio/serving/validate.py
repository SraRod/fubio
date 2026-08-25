"""Submission integrity checks — every failure is loud, none is silent.

Why this exists: `docs/evaluation.md` states that missing results are penalised
with the *worst score among valid submissions*. A silently dropped image is
therefore not "one image omitted", it is one image scored as the worst model in
the field. Before this module, an unreadable file, an absent task directory, or a
mistyped data root all produced a smaller-but-plausible JSON and a zip a human
would upload without noticing.

Two entry points, deliberately not one, because the two representations differ.
In-memory results carry `submission_path` and `original_hw`; the serialized
document renames the first to `image_path` and drops the second entirely, so it
cannot check pixel bounds and must be given the expected index instead.

    validate_inference_results  — model output, before anything is written
    validate_submission_document — the exact bytes about to be zipped

Upstream: serving/predict.py (results), scripts/make_submission.py (document).
Downstream: nothing — these raise or return None.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fubio.data.task_registry import TASKS

# Coordinates are rounded to 6 (normalized) / 2 (pixel) decimals before writing,
# so bounds checks need to tolerate that quantisation and nothing more.
_NORM_TOL = 1e-4
_PIXEL_TOL = 0.51

_REQUIRED_FIELDS = (
    "image_path",
    "task_id",
    "predicted_points_normalized",
    "predicted_points_pixels",
)


class SubmissionError(RuntimeError):
    """A submission artifact is incomplete, malformed, or inconsistent."""


def _expected_len(task_id: str) -> int:
    if task_id not in TASKS:
        raise SubmissionError(f"Unknown task_id {task_id!r}; expected one of {sorted(TASKS)}")
    return 2 * TASKS[task_id].n_keypoints


def _check_finite(values: Sequence[float], label: str) -> None:
    for i, v in enumerate(values):
        if not isinstance(v, int | float) or not math.isfinite(float(v)):
            raise SubmissionError(f"{label}: non-finite coordinate at index {i}: {v!r}")


def _check_vector_lengths(entry: Mapping[str, Any], label: str) -> None:
    task_id = entry["task_id"]
    want = _expected_len(task_id)
    for field in ("predicted_points_pixels", "predicted_points_normalized"):
        got = len(entry[field])
        if got != want:
            raise SubmissionError(
                f"{label}: {field} has {got} values, expected {want} "
                f"(task {task_id} has {TASKS[task_id].n_keypoints} landmarks)"
            )
        _check_finite(entry[field], f"{label}.{field}")


def _check_normalized_range(entry: Mapping[str, Any], label: str) -> None:
    for i, v in enumerate(entry["predicted_points_normalized"]):
        if not (-_NORM_TOL <= float(v) <= 1.0 + _NORM_TOL):
            raise SubmissionError(
                f"{label}: normalized coordinate {i} = {v} outside [0, 1]. "
                f"The organizer spec requires normalized coordinates in [0, 1]."
            )


def _index_of(entries: Iterable[Mapping[str, Any]], path_key: str) -> set[tuple[str, str]]:
    """(task_id, path) identity set, rejecting duplicates."""
    seen: set[tuple[str, str]] = set()
    for e in entries:
        key = (str(e["task_id"]), str(e[path_key]))
        if key in seen:
            raise SubmissionError(f"Duplicate prediction for {key[0]}/{key[1]}")
        seen.add(key)
    return seen


def validate_inference_results(
    samples: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    required_tasks: set[str] | None = None,
) -> None:
    """Model output must cover every discovered sample exactly once.

    Args:
        samples: what discovery found — each needs `task_id`, `submission_path`.
        results: what inference produced — the dicts `save_outputs` will write,
            each additionally carrying `original_hw` as (H, W).
        required_tasks: task ids the caller demands be present. This is a
            **caller policy, not a property of a valid submission**: the local
            competition run wants all nine, but a CSV-driven Docker input may
            legitimately contain a subset. Pass None to skip the check.

    Raises:
        SubmissionError: on any count, identity, schema, or range violation.
    """
    if len(results) != len(samples):
        raise SubmissionError(
            f"Prediction count {len(results)} != discovered sample count {len(samples)}. "
            f"Predictions are missing for {len(samples) - len(results)} input(s); "
            f"the platform scores missing results as the worst valid submission."
        )

    want = _index_of(samples, "submission_path")
    got = _index_of(results, "submission_path")
    if want != got:
        missing = sorted(want - got)
        extra = sorted(got - want)
        raise SubmissionError(
            f"Prediction identities do not match discovered inputs. "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"unexpected={extra[:5]}{'...' if len(extra) > 5 else ''}"
        )

    for entry in results:
        label = f"{entry['task_id']}/{entry['submission_path']}"
        _check_vector_lengths(entry, label)
        _check_normalized_range(entry, label)

        oh, ow = int(entry["original_hw"][0]), int(entry["original_hw"][1])
        px = entry["predicted_points_pixels"]
        for i in range(0, len(px), 2):
            x, y = float(px[i]), float(px[i + 1])
            if not (-_PIXEL_TOL <= x <= ow - 1 + _PIXEL_TOL):
                raise SubmissionError(f"{label}: x={x} outside [0, {ow - 1}]")
            if not (-_PIXEL_TOL <= y <= oh - 1 + _PIXEL_TOL):
                raise SubmissionError(f"{label}: y={y} outside [0, {oh - 1}]")

    if required_tasks is not None:
        present = {str(e["task_id"]) for e in results}
        absent = required_tasks - present
        if absent:
            raise SubmissionError(
                f"No predictions for required task(s): {sorted(absent)}. "
                f"Either the data root is wrong or images are missing."
            )


def validate_submission_document(
    document: Any,
    expected_index: set[tuple[str, str]],
) -> None:
    """Validate the parsed JSON that is about to be zipped.

    Checks the serialized form rather than the in-memory one — that is the thing
    a human actually uploads, and it is a different object from what
    `validate_inference_results` saw.

    Args:
        document: the parsed `regression_predictions.json` (a list of dicts).
        expected_index: {(task_id, image_path)} the document must cover exactly.

    Raises:
        SubmissionError: on any structural or coverage violation.
    """
    if not isinstance(document, list):
        raise SubmissionError(f"Submission must be a JSON list, got {type(document).__name__}")
    if not document:
        raise SubmissionError("Submission is empty")

    for i, entry in enumerate(document):
        if not isinstance(entry, dict):
            raise SubmissionError(f"Entry {i} is {type(entry).__name__}, expected an object")
        missing = [f for f in _REQUIRED_FIELDS if f not in entry]
        if missing:
            raise SubmissionError(f"Entry {i} missing required field(s): {missing}")

        label = f"entry[{i}] {entry['task_id']}/{entry['image_path']}"
        _check_vector_lengths(entry, label)
        _check_normalized_range(entry, label)

    got = _index_of(document, "image_path")
    if got != expected_index:
        missing = sorted(expected_index - got)
        extra = sorted(got - expected_index)
        raise SubmissionError(
            f"Serialized submission does not match the expected inputs. "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"unexpected={extra[:5]}{'...' if len(extra) > 5 else ''}"
        )
