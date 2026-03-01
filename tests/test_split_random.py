import numpy as np
import pandas as pd
import pytest

from splitsmith import split


def test_deterministic_same_seed():
    df = pd.DataFrame({"x": range(100), "y": [0] * 80 + [1] * 20})
    r1 = split(df, target="y", seed=42)
    r2 = split(df, target="y", seed=42)
    assert np.array_equal(r1.train_idx, r2.train_idx)
    assert np.array_equal(r1.val_idx, r2.val_idx)
    assert np.array_equal(r1.test_idx, r2.test_idx)


def test_deterministic_diff_seed():
    df = pd.DataFrame({"x": range(100), "y": [0] * 80 + [1] * 20})
    r1 = split(df, target="y", seed=1)
    r2 = split(df, target="y", seed=2)
    # Very likely different; assert not all equal
    assert not (
        np.array_equal(r1.train_idx, r2.train_idx)
        and np.array_equal(r1.val_idx, r2.val_idx)
        and np.array_equal(r1.test_idx, r2.test_idx)
    )


def test_no_overlap_and_full_coverage():
    df = pd.DataFrame({"x": range(200), "y": [0] * 160 + [1] * 40})
    r = split(df, target="y", seed=42)

    train = set(r.train_idx.tolist())
    val = set(r.val_idx.tolist())
    test = set(r.test_idx.tolist())

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    union = train | val | test
    assert len(union) == len(df)
    assert min(union) == 0
    assert max(union) == len(df) - 1


def test_sizes_sum_and_minimums():
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    r = split(df, target="y", seed=0, stratify=False, ratios=(0.7, 0.2, 0.1))

    assert len(r.train_idx) + len(r.val_idx) + len(r.test_idx) == 10
    assert len(r.train_idx) >= 1
    assert len(r.val_idx) >= 1
    assert len(r.test_idx) >= 1


def test_stratification_preserves_classes():
    df = pd.DataFrame({"x": range(100), "y": [0] * 80 + [1] * 20})
    r = split(df, target="y", seed=42)  # stratify auto

    y = df["y"]
    overall_p1 = (y == 1).mean()
    p1_train = (y.iloc[r.train_idx] == 1).mean()
    p1_val = (y.iloc[r.val_idx] == 1).mean()
    p1_test = (y.iloc[r.test_idx] == 1).mean()

    # each split should have both classes
    assert set(y.iloc[r.train_idx].unique()) == {0, 1}
    assert set(y.iloc[r.val_idx].unique()) == {0, 1}
    assert set(y.iloc[r.test_idx].unique()) == {0, 1}

    # proportions roughly match overall
    assert abs(p1_train - overall_p1) <= 0.1
    assert abs(p1_val - overall_p1) <= 0.1
    assert abs(p1_test - overall_p1) <= 0.1


def test_stratification_too_few_samples_raises():
    df = pd.DataFrame({"x": range(10), "y": [0] * 8 + [1] * 2})
    with pytest.raises(ValueError):
        split(df, target="y", seed=0, stratify=True)


def test_strategy_not_implemented():
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    with pytest.raises(NotImplementedError):
        split(df, target="y", strategy="time")


def test_groups_timecol_not_implemented():
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5, "t": range(10), "g": list("aaaaaaaaaa")})
    with pytest.raises(NotImplementedError):
        split(df, target="y", groups="g")
    with pytest.raises(NotImplementedError):
        split(df, target="y", time_col="t")