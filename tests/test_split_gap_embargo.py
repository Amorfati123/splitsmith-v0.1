"""Tests for gap/embargo support in time and group_time splits."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import split, audit


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


class TestTimeGapEmbargo:
    def test_gap_creates_separation(self):
        df = _time_df()
        r = split(df, target="target", strategy="time", time_col="timestamp", gap=5)
        train_max = df.iloc[r.train_idx]["timestamp"].max()
        val_min = df.iloc[r.val_idx]["timestamp"].min()
        # With gap=5, there should be dropped rows between train and val
        assert train_max < val_min

    def test_gap_metadata(self):
        df = _time_df()
        r = split(df, target="target", strategy="time", time_col="timestamp", gap=3)
        assert r.metadata["gap"] == 3
        assert r.metadata["n_dropped_gap"] == 6  # 2 * gap

    def test_gap_reduces_usable_rows(self):
        df = _time_df()
        r_no_gap = split(df, target="target", strategy="time", time_col="timestamp", gap=0)
        r_gap = split(df, target="target", strategy="time", time_col="timestamp", gap=5)
        total_no_gap = len(r_no_gap.train_idx) + len(r_no_gap.val_idx) + len(r_no_gap.test_idx)
        total_gap = len(r_gap.train_idx) + len(r_gap.val_idx) + len(r_gap.test_idx)
        assert total_gap < total_no_gap

    def test_embargo_alias(self):
        df = _time_df()
        r_gap = split(df, target="target", strategy="time", time_col="timestamp", gap=5)
        r_embargo = split(df, target="target", strategy="time", time_col="timestamp", embargo=5)
        assert np.array_equal(r_gap.train_idx, r_embargo.train_idx)
        assert np.array_equal(r_gap.val_idx, r_embargo.val_idx)

    def test_gap_and_embargo_takes_max(self):
        df = _time_df()
        r = split(df, target="target", strategy="time", time_col="timestamp", gap=3, embargo=7)
        assert r.metadata["gap"] == 7

    def test_too_large_gap_raises(self):
        df = _time_df(n=10)
        with pytest.raises(ValueError, match="only .* rows remain"):
            split(df, target="target", strategy="time", time_col="timestamp", gap=5)

    def test_gap_zero_is_default(self):
        df = _time_df()
        r = split(df, target="target", strategy="time", time_col="timestamp")
        assert r.metadata["gap"] == 0

    def test_negative_gap_raises(self):
        df = _time_df()
        with pytest.raises(ValueError, match="non-negative"):
            split(df, target="target", strategy="time", time_col="timestamp", gap=-1)

    def test_audit_clean_with_gap(self):
        df = _time_df()
        r = split(df, target="target", strategy="time", time_col="timestamp", gap=5)
        report = audit(df, r, "target", time_col="timestamp")
        assert report.ok is True


class TestGroupTimeGapEmbargo:
    def test_gap_creates_group_separation(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time",
                  groups="group_id", time_col="timestamp", gap=2)
        train_max = df.iloc[r.train_idx]["timestamp"].max()
        val_min = df.iloc[r.val_idx]["timestamp"].min()
        assert train_max < val_min

    def test_gap_metadata(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time",
                  groups="group_id", time_col="timestamp", gap=2)
        assert r.metadata["gap"] == 2
        assert r.metadata["n_dropped_gap_groups"] == 4  # 2 * gap

    def test_too_large_gap_raises(self):
        df = _group_time_df(n_groups=4)
        with pytest.raises(ValueError, match="only .* groups remain"):
            split(df, target="target", strategy="group_time",
                  groups="group_id", time_col="timestamp", gap=2)

    def test_audit_clean_with_gap(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time",
                  groups="group_id", time_col="timestamp", gap=2)
        report = audit(df, r, "target", groups="group_id", time_col="timestamp")
        assert report.ok is True
