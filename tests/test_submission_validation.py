"""The submission path must fail loudly rather than emit a smaller file.

`docs/evaluation.md`: "Missing results penalized with worst score among valid
submissions." So a dropped image is not an omission, it is that image scored as
the worst model in the field. Before these checks, three separate silent paths
could shrink a submission — an absent task directory, an unreadable image, and a
batch in which every image failed — and `make_submission.py` verified only that
the JSON file existed.

Each test below names the specific way a submission could have gone out wrong.
"""

from __future__ import annotations

import json
import math

import pytest

from fubio.data.task_registry import TASKS
from fubio.serving.validate import (
    SubmissionError,
    validate_inference_results,
    validate_submission_document,
)

# IVC is the smallest task (2 landmarks -> 4 values); HC has 4 (-> 8).
_IVC_K = TASKS["IVC"].n_keypoints
_HC_K = TASKS["HC"].n_keypoints


def _sample(task_id: str, name: str) -> dict:
    return {
        "task_id": task_id,
        "submission_path": f"{task_id}/{name}",
        "image_path": f"val/{task_id}/{name}",
    }


def _result(task_id: str, name: str, *, h: int = 100, w: int = 200) -> dict:
    k = TASKS[task_id].n_keypoints
    px, norm = [], []
    for i in range(k):
        x, y = float(10 + i), float(20 + i)
        px += [x, y]
        norm += [x / w, y / h]
    return {
        "task_id": task_id,
        "submission_path": f"{task_id}/{name}",
        "image_path": f"val/{task_id}/{name}",
        "predicted_points_pixels": px,
        "predicted_points_normalized": norm,
        "original_hw": [h, w],
    }


def _pair(specs: list[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    return [_sample(t, n) for t, n in specs], [_result(t, n) for t, n in specs]


class TestInferenceResults:
    def test_happy_path_all_tasks(self) -> None:
        specs = [(t, f"{t}_0.png") for t in TASKS]
        samples, results = _pair(specs)
        validate_inference_results(samples, results, required_tasks=set(TASKS))

    def test_count_mismatch_raises(self) -> None:
        """The core failure: inference silently produced fewer predictions."""
        samples, results = _pair([("IVC", "a.png"), ("HC", "b.png")])
        with pytest.raises(SubmissionError, match="count"):
            validate_inference_results(samples, results[:-1], required_tasks=None)

    def test_identity_mismatch_raises(self) -> None:
        """Right count, wrong images — a count check alone would pass this."""
        samples, results = _pair([("IVC", "a.png"), ("HC", "b.png")])
        results[1]["submission_path"] = "HC/WRONG.png"
        with pytest.raises(SubmissionError, match="do not match"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_duplicate_prediction_raises(self) -> None:
        samples, results = _pair([("IVC", "a.png"), ("HC", "b.png")])
        results[1] = _result("IVC", "a.png")
        with pytest.raises(SubmissionError, match="Duplicate"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_wrong_vector_length_raises(self) -> None:
        samples, results = _pair([("IVC", "a.png")])
        results[0]["predicted_points_pixels"] = [1.0, 2.0]  # IVC needs 4
        with pytest.raises(SubmissionError, match="expected"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_nan_coordinate_raises(self) -> None:
        samples, results = _pair([("IVC", "a.png")])
        results[0]["predicted_points_pixels"][0] = math.nan
        with pytest.raises(SubmissionError, match="non-finite"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_inf_coordinate_raises(self) -> None:
        samples, results = _pair([("IVC", "a.png")])
        results[0]["predicted_points_pixels"][0] = math.inf
        with pytest.raises(SubmissionError, match="non-finite"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_normalized_out_of_range_raises(self) -> None:
        samples, results = _pair([("IVC", "a.png")])
        results[0]["predicted_points_normalized"][0] = 1.5
        with pytest.raises(SubmissionError, match="outside"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_pixel_outside_image_raises(self) -> None:
        """Catches a bypass of the official inverse, or swapped H/W."""
        samples, results = _pair([("IVC", "a.png")])
        results[0]["predicted_points_pixels"][0] = 9999.0
        with pytest.raises(SubmissionError, match="outside"):
            validate_inference_results(samples, results, required_tasks=None)

    def test_missing_required_task_raises(self) -> None:
        samples, results = _pair([("IVC", "a.png")])
        with pytest.raises(SubmissionError, match="required task"):
            validate_inference_results(samples, results, required_tasks=set(TASKS))

    def test_required_tasks_is_a_caller_policy(self) -> None:
        """A task subset is legal when the caller says so.

        Kept caller-supplied rather than baked in because the Docker input is
        CSV-driven and may legitimately carry a subset — a validator that
        hardcoded "all nine" would need rewriting at that point.
        """
        samples, results = _pair([("IVC", "a.png")])
        validate_inference_results(samples, results, required_tasks={"IVC"})
        validate_inference_results(samples, results, required_tasks=None)


class TestSubmissionDocument:
    def _doc(self, specs: list[tuple[str, str]]) -> tuple[list[dict], set[tuple[str, str]]]:
        results = [_result(t, n) for t, n in specs]
        doc = [
            {
                "image_path": r["submission_path"],
                "task_id": r["task_id"],
                "predicted_points_normalized": r["predicted_points_normalized"],
                "predicted_points_pixels": r["predicted_points_pixels"],
            }
            for r in results
        ]
        return doc, {(d["task_id"], d["image_path"]) for d in doc}

    def test_happy_path(self) -> None:
        doc, index = self._doc([("IVC", "a.png"), ("HC", "b.png")])
        validate_submission_document(doc, index)

    def test_not_a_list_raises(self) -> None:
        with pytest.raises(SubmissionError, match="JSON list"):
            validate_submission_document({"nope": 1}, set())

    def test_empty_raises(self) -> None:
        with pytest.raises(SubmissionError, match="empty"):
            validate_submission_document([], set())

    def test_missing_field_raises(self) -> None:
        doc, index = self._doc([("IVC", "a.png")])
        del doc[0]["predicted_points_pixels"]
        with pytest.raises(SubmissionError, match="missing required field"):
            validate_submission_document(doc, index)

    def test_altered_path_raises(self) -> None:
        """Corruption between write and zip — the state a human uploads."""
        doc, index = self._doc([("IVC", "a.png"), ("HC", "b.png")])
        doc[0]["image_path"] = "IVC/tampered.png"
        with pytest.raises(SubmissionError, match="does not match"):
            validate_submission_document(doc, index)

    def test_dropped_entry_raises(self) -> None:
        doc, index = self._doc([("IVC", "a.png"), ("HC", "b.png")])
        with pytest.raises(SubmissionError, match="does not match"):
            validate_submission_document(doc[:1], index)

    def test_unknown_task_raises(self) -> None:
        doc, index = self._doc([("IVC", "a.png")])
        doc[0]["task_id"] = "NOT_A_TASK"
        with pytest.raises(SubmissionError, match="Unknown task_id"):
            validate_submission_document(doc, {("NOT_A_TASK", "IVC/a.png")})


class TestStrictJson:
    def test_json_dump_rejects_nan(self, tmp_path) -> None:
        """Python emits a bare `NaN` token by default, which is not valid JSON.

        Whether the platform's parser accepts it is unknown, and with one final
        submission that is not a question worth leaving open.
        """
        with pytest.raises(ValueError):
            json.dumps([{"x": math.nan}], allow_nan=False)
        # and the default really does produce the invalid token
        assert "NaN" in json.dumps([{"x": math.nan}])
