"""Tests for gap/embargo in time-aware cross-validation."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import k_fold
from splitsmith.audit import audit_cv


def _time_df(n=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


def _group_time_df(n_groups=15, rows_per_group=5):
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({
                "group_id": f"g{i}",
                "timestamp": base + pd.Timedelta(days=i * rows_per_group + j),
                "feature": i * rows_per_group + j,
                "target": j % 2,
            })
    return pd.DataFrame(rows)


class TestTimeCVGap:
    def test_gap_shrinks_train(self):
        """With gap>0, each fold's train set should be smaller."""
        df = _time_df()
        cv_no_gap = k_fold(df, target="target", k=3, strategy="time", time_col="timestamp", gap=0)
        cv_gap = k_fold(df, target="target", k=3, strategy="time", time_col="timestamp", gap=5)
        for i in range(3):
            assert len(cv_gap.folds[i].train_idx) <= len(cv_no_gap.folds[i].train_idx)

    def test_gap_preserves_temporal_order(self):
        """Train max should be before val min in each fold with gap."""
        df = _time_df()
        cv = k_fold(df, target="target", k=3, strategy="time", time_col="timestamp", gap=3)
        for fold in cv.folds:
            train_max = df.iloc[fold.train_idx]["timestamp"].max()
            val_min = df.iloc[fold.val_idx]["timestamp"].min()
            assert train_max < val_min

    def test_gap_in_metadata(self):
        df = _time_df()
        cv = k_fold(df, target="target", k=3, strategy="time", time_col="timestamp", gap=5)
        assert cv.metadata["gap"] == 5
        for fold in cv.folds:
            assert fold.metadata["gap"] == 5

    def test_embargo_alias(self):
        df = _time_df()
        cv = k_fold(df, target="target", k=3, strategy="time", time_col="timestamp", embargo=5)
        assert cv.metadata["gap"] == 5

    def test_negative_gap_raises(self):
        df = _time_df()
        with pytest.raises(ValueError, match="non-negative"):
            k_fold(df, target="target", k=3, strategy="time", time_col="timestamp", gap=-1)


class TestGroupTimeCVGap:
    def test_gap_shrinks_train(self):
        df = _group_time_df()
        cv_no_gap = k_fold(df, target="target", k=3, strategy="group_time",
                           groups="group_id", time_col="timestamp", gap=0)
        cv_gap = k_fold(df, target="target", k=3, strategy="group_time",
                        groups="group_id", time_col="timestamp", gap=1)
        for i in range(3):
            assert len(cv_gap.folds[i].train_idx) <= len(cv_no_gap.folds[i].train_idx)

    def test_gap_preserves_temporal_order(self):
        df = _group_time_df()
        cv = k_fold(df, target="target", k=3, strategy="group_time",
                    groups="group_id", time_col="timestamp", gap=1)
        for fold in cv.folds:
            train_max = df.iloc[fold.train_idx]["timestamp"].max()
            val_min = df.iloc[fold.val_idx]["timestamp"].min()
            assert train_max < val_min

    def test_gap_in_metadata(self):
        df = _group_time_df()
        cv = k_fold(df, target="target", k=3, strategy="group_time",
                    groups="group_id", time_col="timestamp", gap=2)
        assert cv.metadata["gap"] == 2
