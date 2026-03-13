"""Leakage auditing for splitsmith."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .types import CVResult, Finding, LeakageReport, SplitResult

_MAX_EVIDENCE_ROWS = 5


# ---------------------------------------------------------------------------
# Safe hashing for DataFrames with unhashable columns
# ---------------------------------------------------------------------------

def _safe_serialize(val: Any) -> str:
    """Convert a single cell value to a deterministic string for hashing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "__NaN__"
    if isinstance(val, np.ndarray):
        return f"ndarray:{val.tobytes().hex()}"
    if isinstance(val, (list, tuple)):
        try:
            return json.dumps(val, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(val)
    if isinstance(val, dict):
        try:
            return json.dumps(val, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def _has_unhashable_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
    """Check whether any column in *columns* (or all columns) contains unhashable types."""
    cols = columns if columns is not None else df.columns.tolist()
    for col in cols:
        try:
            pd.util.hash_pandas_object(df[[col]], index=False)
        except (TypeError, ValueError):
            return True
    return False


def _hash_rows(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    unhashable_policy: str = "serialize",
) -> pd.Series:
    """Return a Series of per-row hashes, handling unhashable columns gracefully.

    Parameters
    ----------
    columns : list of column names to include (None = all columns).
    unhashable_policy : "serialize" | "skip" | "error"
        - serialize: convert unhashable cells to deterministic strings
        - skip: drop columns that fail native hashing
        - error: raise on unhashable columns
    """
    sub = df[columns] if columns is not None else df

    # Fast path: try native pandas hashing first
    try:
        return pd.util.hash_pandas_object(sub, index=False)
    except (TypeError, ValueError):
        pass

    if unhashable_policy == "error":
        raise TypeError(
            "DataFrame contains unhashable column types. "
            "Use ignore_columns or duplicate_subset to restrict checked columns, "
            "or set unhashable_policy='serialize' or 'skip'."
        )

    if unhashable_policy == "skip":
        safe_cols = []
        for col in sub.columns:
            try:
                pd.util.hash_pandas_object(sub[[col]], index=False)
                safe_cols.append(col)
            except (TypeError, ValueError):
                pass
        if not safe_cols:
            # Nothing is hashable — return unique hashes so no duplicates are found
            return pd.Series(range(len(sub)), index=sub.index)
        return pd.util.hash_pandas_object(sub[safe_cols], index=False)

    # Default: serialize
    hashes = []
    for idx in range(len(sub)):
        row_str = "|".join(_safe_serialize(sub.iloc[idx, c]) for c in range(sub.shape[1]))
        h = int(hashlib.sha256(row_str.encode("utf-8")).hexdigest()[:16], 16)
        hashes.append(h)
    return pd.Series(hashes, index=sub.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit(
    df: pd.DataFrame,
    split_result: SplitResult,
    target: str,
    *,
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    ignore_columns: Optional[Sequence[str]] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
    unhashable_policy: str = "serialize",
) -> LeakageReport:
    """Audit a split for leakage and integrity issues.

    Parameters
    ----------
    df : DataFrame
    split_result : SplitResult from split()
    target : target column name
    groups : optional group column for group-leakage check
    time_col : optional time column for time-leakage check
    ignore_columns : columns to exclude from duplicate detection
    duplicate_subset : columns to use for duplicate detection (overrides ignore_columns)
    unhashable_policy : "serialize" | "skip" | "error"
        How to handle columns containing arrays/dicts/lists.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(split_result, SplitResult):
        raise TypeError("split_result must be a SplitResult")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if groups is not None and groups not in df.columns:
        raise ValueError(f"groups column '{groups}' not found in DataFrame")
    if time_col is not None and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")
    if unhashable_policy not in ("serialize", "skip", "error"):
        raise ValueError(f"unhashable_policy must be 'serialize', 'skip', or 'error', got {unhashable_policy!r}")

    # Resolve columns for duplicate checking
    dup_columns = _resolve_dup_columns(df, ignore_columns, duplicate_subset)

    report = LeakageReport()
    _check_overlap(split_result, report)
    _check_duplicates(df, split_result, report, dup_columns, unhashable_policy)
    if groups is not None:
        _check_group_leakage(df, split_result, groups, report)
    if time_col is not None:
        _check_time_leakage(df, split_result, time_col, report)
    return report


def _resolve_dup_columns(
    df: pd.DataFrame,
    ignore_columns: Optional[Sequence[str]],
    duplicate_subset: Optional[Sequence[str]],
) -> Optional[List[str]]:
    """Return the list of columns for duplicate checking, or None for all columns."""
    if duplicate_subset is not None:
        cols = list(duplicate_subset)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"duplicate_subset columns not found in DataFrame: {missing}")
        return cols
    if ignore_columns is not None:
        ignore = set(ignore_columns)
        missing = [c for c in ignore if c not in df.columns]
        if missing:
            raise ValueError(f"ignore_columns not found in DataFrame: {missing}")
        return [c for c in df.columns if c not in ignore]
    return None


def audit_cv(
    df: pd.DataFrame,
    cv_result: CVResult,
    target: str,
    *,
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    ignore_columns: Optional[Sequence[str]] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
    unhashable_policy: str = "serialize",
) -> List[LeakageReport]:
    """Audit every fold of a CV result for leakage."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(cv_result, CVResult):
        raise TypeError("cv_result must be a CVResult")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if groups is not None and groups not in df.columns:
        raise ValueError(f"groups column '{groups}' not found in DataFrame")
    if time_col is not None and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")

    reports = []
    for fold in cv_result.folds:
        sr = SplitResult(
            train_idx=fold.train_idx,
            val_idx=fold.val_idx,
            test_idx=np.array([], dtype=int),
        )
        reports.append(audit(
            df, sr, target,
            groups=groups, time_col=time_col,
            ignore_columns=ignore_columns,
            duplicate_subset=duplicate_subset,
            unhashable_policy=unhashable_policy,
        ))
    return reports


def audit_cv_summary(reports: List[LeakageReport]) -> Dict[str, Any]:
    """Summarize audit results across all folds."""
    return {
        "n_folds": len(reports),
        "all_ok": all(r.ok for r in reports),
        "per_fold": [r.summary() for r in reports],
        "total_errors": sum(r.summary()["error"] for r in reports),
        "total_warnings": sum(r.summary()["warn"] for r in reports),
    }


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def _check_overlap(sr: SplitResult, report: LeakageReport) -> None:
    splits = {
        "train": set(sr.train_idx.tolist()),
        "val": set(sr.val_idx.tolist()),
        "test": set(sr.test_idx.tolist()),
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    found_overlap = False

    for a, b in pairs:
        overlap = splits[a] & splits[b]
        if overlap:
            found_overlap = True
            examples = sorted(overlap)[:_MAX_EVIDENCE_ROWS]
            report.add(Finding(
                id="index_overlap", severity="error",
                title=f"Index overlap: {a} ∩ {b}",
                details=f"{len(overlap)} index(es) appear in both {a} and {b}.",
                evidence={"count": len(overlap), "examples": examples},
            ))

    if not found_overlap:
        report.add(Finding(
            id="index_overlap", severity="info",
            title="No index overlap",
            details="All split indices are mutually exclusive.",
        ))


def _check_duplicates(
    df: pd.DataFrame,
    sr: SplitResult,
    report: LeakageReport,
    columns: Optional[List[str]] = None,
    unhashable_policy: str = "serialize",
) -> None:
    idx_to_split: Dict[int, str] = {}
    for idx in sr.train_idx.tolist():
        idx_to_split[idx] = "train"
    for idx in sr.val_idx.tolist():
        idx_to_split[idx] = "val"
    for idx in sr.test_idx.tolist():
        idx_to_split[idx] = "test"

    row_hashes = _hash_rows(df, columns=columns, unhashable_policy=unhashable_policy)
    hash_to_indices: Dict[int, List[int]] = {}
    for idx, h in row_hashes.items():
        hash_to_indices.setdefault(int(h), []).append(int(idx))

    dup_groups = [indices for indices in hash_to_indices.values() if len(indices) > 1]

    if not dup_groups:
        report.add(Finding(
            id="duplicate_rows", severity="info",
            title="No duplicate rows",
            details="All rows across all splits are unique.",
        ))
        return

    cross_split: List[List[int]] = []
    within_split: List[List[int]] = []

    for group_indices in dup_groups:
        split_names = {idx_to_split[i] for i in group_indices if i in idx_to_split}
        if len(split_names) > 1:
            cross_split.append(group_indices)
        elif len(split_names) == 1:
            within_split.append(group_indices)

    if cross_split:
        examples = []
        for gi in cross_split[:_MAX_EVIDENCE_ROWS]:
            splits = sorted({idx_to_split[i] for i in gi if i in idx_to_split})
            examples.append({"indices": gi[:_MAX_EVIDENCE_ROWS], "splits": splits})
        report.add(Finding(
            id="duplicate_rows", severity="error",
            title="Cross-split duplicate rows",
            details=f"{len(cross_split)} group(s) of identical rows span multiple splits.",
            evidence={"n_groups": len(cross_split), "n_rows": sum(len(g) for g in cross_split), "examples": examples},
        ))

    if within_split:
        examples = []
        for gi in within_split[:_MAX_EVIDENCE_ROWS]:
            splits = sorted({idx_to_split[i] for i in gi if i in idx_to_split})
            examples.append({"indices": gi[:_MAX_EVIDENCE_ROWS], "splits": splits})
        report.add(Finding(
            id="duplicate_rows", severity="warn",
            title="Within-split duplicate rows",
            details=f"{len(within_split)} group(s) of identical rows found within the same split.",
            evidence={"n_groups": len(within_split), "n_rows": sum(len(g) for g in within_split), "examples": examples},
        ))


def _check_group_leakage(df: pd.DataFrame, sr: SplitResult, groups: str, report: LeakageReport) -> None:
    train_groups = set(df.iloc[sr.train_idx][groups].unique())
    val_groups = set(df.iloc[sr.val_idx][groups].unique())
    test_groups = set(df.iloc[sr.test_idx][groups].unique())

    pairs = [
        ("train", "val", train_groups, val_groups),
        ("train", "test", train_groups, test_groups),
        ("val", "test", val_groups, test_groups),
    ]
    found = False
    for a_name, b_name, a_set, b_set in pairs:
        shared = a_set & b_set
        if shared:
            found = True
            examples = sorted(shared, key=str)[:_MAX_EVIDENCE_ROWS]
            report.add(Finding(
                id="group_leakage", severity="error",
                title=f"Group leakage: {a_name} ∩ {b_name}",
                details=f"{len(shared)} group(s) appear in both {a_name} and {b_name}.",
                evidence={"count": len(shared), "examples": examples},
            ))
    if not found:
        report.add(Finding(
            id="group_leakage", severity="info",
            title=f"No group leakage on '{groups}'",
            details="Each group value appears in exactly one split.",
        ))


def _check_time_leakage(df: pd.DataFrame, sr: SplitResult, time_col: str, report: LeakageReport) -> None:
    ts = pd.to_datetime(df[time_col])
    train_ts = ts.iloc[sr.train_idx]
    val_ts = ts.iloc[sr.val_idx]
    test_ts = ts.iloc[sr.test_idx]

    train_max = train_ts.max()
    val_min = val_ts.min()
    val_max = val_ts.max()
    test_min = test_ts.min()

    issues_found = False

    if train_max > val_min:
        n_v = int((train_ts > val_min).sum())
        report.add(Finding(
            id="time_leakage", severity="error",
            title="Time leakage: train → val",
            details=f"Train max ({train_max.isoformat()}) > val min ({val_min.isoformat()}).",
            evidence={"pair": "train → val", "train_max": train_max.isoformat(), "val_min": val_min.isoformat(), "n_violating_rows": n_v},
        ))
        issues_found = True

    if train_max > test_min:
        n_v = int((train_ts > test_min).sum())
        report.add(Finding(
            id="time_leakage", severity="error",
            title="Time leakage: train → test",
            details=f"Train max ({train_max.isoformat()}) > test min ({test_min.isoformat()}).",
            evidence={"pair": "train → test", "train_max": train_max.isoformat(), "test_min": test_min.isoformat(), "n_violating_rows": n_v},
        ))
        issues_found = True

    if val_max > test_min:
        n_v = int((val_ts > test_min).sum())
        report.add(Finding(
            id="time_leakage", severity="warn",
            title="Time leakage: val → test",
            details=f"Val max ({val_max.isoformat()}) > test min ({test_min.isoformat()}).",
            evidence={"pair": "val → test", "val_max": val_max.isoformat(), "test_min": test_min.isoformat(), "n_violating_rows": n_v},
        ))
        issues_found = True

    if not issues_found:
        report.add(Finding(
            id="time_leakage", severity="info",
            title=f"No time leakage on '{time_col}'",
            details="Temporal ordering is clean: train < val < test.",
        ))
