"""Tests for audit() handling of unhashable/array-valued columns."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import SplitResult, audit


def _make_result(train, val, test):
    return SplitResult(
        train_idx=np.array(train, dtype=int),
        val_idx=np.array(val, dtype=int),
        test_idx=np.array(test, dtype=int),
    )


class TestUnhashableColumns:
    """Audit must not crash on DataFrames with array/list/dict columns."""

    def test_array_column_serialize_policy(self):
        """Default serialize policy handles numpy array columns."""
        df = pd.DataFrame({
            "feature": [np.array([1, 2, 3]), np.array([4, 5, 6]),
                         np.array([7, 8, 9]), np.array([10, 11, 12]),
                         np.array([13, 14, 15]), np.array([16, 17, 18])],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        assert report.ok is True

    def test_list_column_serialize_policy(self):
        """Serialize policy handles list-valued columns."""
        df = pd.DataFrame({
            "embedding": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        assert report.ok is True

    def test_dict_column_serialize_policy(self):
        """Serialize policy handles dict-valued columns."""
        df = pd.DataFrame({
            "metadata": [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}, {"f": 6}],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        assert report.ok is True

    def test_array_duplicates_detected_serialize(self):
        """Duplicate array rows across splits should be caught."""
        arr = np.array([1, 2, 3])
        df = pd.DataFrame({
            "feature": [arr, np.array([4, 5, 6]), arr, np.array([7, 8, 9]),
                         np.array([10, 11, 12]), np.array([13, 14, 15])],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        errors = [f for f in report.findings if f.id == "duplicate_rows" and f.severity == "error"]
        assert len(errors) == 1

    def test_skip_policy_drops_unhashable_cols(self):
        """Skip policy ignores unhashable columns, checks only scalar ones."""
        df = pd.DataFrame({
            "scalar": [1, 2, 3, 4, 5, 6],
            "array_col": [np.array([1, 2])] * 6,
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", unhashable_policy="skip")
        # All scalar columns are unique, so no duplicates
        assert report.ok is True

    def test_error_policy_raises(self):
        """Error policy raises TypeError on unhashable columns."""
        df = pd.DataFrame({
            "feature": [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]),
                         np.array([7, 8]), np.array([9, 10]), np.array([11, 12])],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        with pytest.raises(TypeError, match="unhashable"):
            audit(df, sr, "target", unhashable_policy="error")

    def test_invalid_policy_rejected(self):
        df = pd.DataFrame({"target": [0, 1, 2, 3, 4, 5]})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        with pytest.raises(ValueError, match="unhashable_policy"):
            audit(df, sr, "target", unhashable_policy="bogus")

    def test_mixed_hashable_unhashable_columns(self):
        """DataFrame with both scalar and array columns works with serialize."""
        df = pd.DataFrame({
            "id": range(6),
            "embedding": [np.array([1.0, 2.0])] * 3 + [np.array([3.0, 4.0])] * 3,
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        # Rows differ by 'id', so no cross-split dups despite same embeddings
        assert report.ok is True


class TestIgnoreColumns:
    """audit(ignore_columns=...) excludes specified columns from duplicate checks."""

    def test_ignore_removes_column_from_check(self):
        """Ignoring the only differing column should reveal duplicates."""
        df = pd.DataFrame({
            "id": range(6),
            "feature": [10, 20, 10, 30, 40, 50],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        # Without ignore: rows 0 and 2 differ by id, so no dups
        report1 = audit(df, sr, "target")
        dup_errors1 = [f for f in report1.findings if f.id == "duplicate_rows" and f.severity == "error"]
        assert len(dup_errors1) == 0

        # Ignoring 'id': rows 0 and 2 are identical on feature+target → cross-split dup
        report2 = audit(df, sr, "target", ignore_columns=["id"])
        dup_errors2 = [f for f in report2.findings if f.id == "duplicate_rows" and f.severity == "error"]
        assert len(dup_errors2) == 1

    def test_ignore_nonexistent_column_raises(self):
        df = pd.DataFrame({"target": [0, 1, 2, 3, 4, 5]})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        with pytest.raises(ValueError, match="ignore_columns"):
            audit(df, sr, "target", ignore_columns=["nonexistent"])

    def test_ignore_unhashable_column(self):
        """Ignoring the array column allows native hashing on remaining cols."""
        df = pd.DataFrame({
            "scalar": [1, 2, 3, 4, 5, 6],
            "array_col": [np.array([1, 2])] * 6,
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        # ignore the array column, check only scalar + target
        report = audit(df, sr, "target", ignore_columns=["array_col"])
        assert report.ok is True


class TestDuplicateSubset:
    """audit(duplicate_subset=...) checks only specified columns for duplicates."""

    def test_subset_narrows_check(self):
        df = pd.DataFrame({
            "id": range(6),
            "feature": [10, 20, 10, 30, 40, 50],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        # Check only 'feature' and 'target': rows 0 and 2 match
        report = audit(df, sr, "target", duplicate_subset=["feature", "target"])
        dup_errors = [f for f in report.findings if f.id == "duplicate_rows" and f.severity == "error"]
        assert len(dup_errors) == 1

    def test_subset_nonexistent_column_raises(self):
        df = pd.DataFrame({"target": [0, 1, 2, 3, 4, 5]})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        with pytest.raises(ValueError, match="duplicate_subset"):
            audit(df, sr, "target", duplicate_subset=["nonexistent"])

    def test_subset_overrides_ignore(self):
        """duplicate_subset takes precedence over ignore_columns."""
        df = pd.DataFrame({
            "id": range(6),
            "feature": [10, 20, 10, 30, 40, 50],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        # Even though ignore_columns=["feature"], subset=["feature","target"] wins
        report = audit(
            df, sr, "target",
            ignore_columns=["feature"],
            duplicate_subset=["feature", "target"],
        )
        dup_errors = [f for f in report.findings if f.id == "duplicate_rows" and f.severity == "error"]
        assert len(dup_errors) == 1
