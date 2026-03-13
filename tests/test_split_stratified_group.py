"""Tests for stratified group splitting."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import split, audit


def _grouped_df(n_groups=10, rows_per_group=10):
    """Create a grouped DataFrame with binary target."""
    rows = []
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({
                "group_id": f"g{i}",
                "feature": i * rows_per_group + j,
                "target": j % 2,
            })
    return pd.DataFrame(rows)


def _imbalanced_grouped_df(n_groups=12, rows_per_group=20):
    """Grouped DataFrame with imbalanced classes (80/20 split)."""
    rows = []
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({
                "group_id": f"g{i}",
                "feature": i * rows_per_group + j,
                "target": 0 if j < int(rows_per_group * 0.8) else 1,
            })
    return pd.DataFrame(rows)


class TestStratifiedGroupSplit:
    def test_groups_are_exclusive(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        train_g = set(df.iloc[r.train_idx]["group_id"].unique())
        val_g = set(df.iloc[r.val_idx]["group_id"].unique())
        test_g = set(df.iloc[r.test_idx]["group_id"].unique())
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)

    def test_all_rows_covered(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        all_idx = set(r.train_idx.tolist()) | set(r.val_idx.tolist()) | set(r.test_idx.tolist())
        assert all_idx == set(range(len(df)))

    def test_deterministic(self):
        df = _grouped_df()
        r1 = split(df, target="target", strategy="group", groups="group_id", stratify=True, seed=42)
        r2 = split(df, target="target", strategy="group", groups="group_id", stratify=True, seed=42)
        assert np.array_equal(r1.train_idx, r2.train_idx)
        assert np.array_equal(r1.val_idx, r2.val_idx)
        assert np.array_equal(r1.test_idx, r2.test_idx)

    def test_class_balance_is_approximate(self):
        """Stratified group split should produce better class balance than plain group."""
        df = _imbalanced_grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        # Overall prevalence of class 1 is 20%
        overall_pct = df["target"].mean()
        train_pct = df.iloc[r.train_idx]["target"].mean()
        val_pct = df.iloc[r.val_idx]["target"].mean()
        test_pct = df.iloc[r.test_idx]["target"].mean()
        # Each split should be within 15pp of overall (generous tolerance for group constraint)
        assert abs(train_pct - overall_pct) < 0.15
        assert abs(val_pct - overall_pct) < 0.15
        assert abs(test_pct - overall_pct) < 0.15

    def test_metadata_has_stratified_flag(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        assert r.metadata["stratified"] is True
        assert "target_counts_train" in r.metadata
        assert "target_counts_val" in r.metadata
        assert "target_counts_test" in r.metadata

    def test_each_split_has_at_least_one_group(self):
        df = _grouped_df(n_groups=3, rows_per_group=5)
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        assert len(r.train_idx) > 0
        assert len(r.val_idx) > 0
        assert len(r.test_idx) > 0

    def test_multiclass_target(self):
        """Works with multiclass targets."""
        rows = []
        for i in range(12):
            for j in range(10):
                rows.append({"group_id": f"g{i}", "feature": i * 10 + j, "target": j % 3})
        df = pd.DataFrame(rows)
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        train_g = set(df.iloc[r.train_idx]["group_id"].unique())
        val_g = set(df.iloc[r.val_idx]["group_id"].unique())
        test_g = set(df.iloc[r.test_idx]["group_id"].unique())
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)

    def test_audit_integration(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id", stratify=True)
        report = audit(df, r, "target", groups="group_id")
        assert report.ok is True

    def test_min_3_groups_required(self):
        df = pd.DataFrame({"group_id": ["A", "A", "B", "B"], "target": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="at least 3"):
            split(df, target="target", strategy="group", groups="group_id", stratify=True)
