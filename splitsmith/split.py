import numpy as np
import pandas as pd

from .types import SplitResult


def _validate_and_normalize_sizes(n: int, ratios) -> tuple[int, int, int]:
    train_size = int(np.floor(ratios[0] * n))
    val_size = int(np.floor(ratios[1] * n))
    test_size = n - train_size - val_size

    sizes = [train_size, val_size, test_size]

    # Ensure each split has at least 1 sample (for n >= 3)
    while any(s == 0 for s in sizes):
        largest = sizes.index(max(sizes))
        smallest = sizes.index(min(sizes))
        if sizes[largest] <= 1:
            raise ValueError("Cannot allocate at least 1 sample to each split with given ratios.")
        sizes[largest] -= 1
        sizes[smallest] += 1

    return sizes[0], sizes[1], sizes[2]


def split(
    df,
    target: str,
    groups=None,
    time_col=None,
    strategy: str = "random",
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
    stratify: bool | None = None,
) -> SplitResult:
    # Basic validations
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found in dataframe")

    if not isinstance(ratios, (tuple, list)) or len(ratios) != 3:
        raise ValueError("ratios must have length 3 (train, val, test)")

    if any(r <= 0 for r in ratios):
        raise ValueError("all ratios must be > 0")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    n = len(df)
    if n < 3:
        raise ValueError("dataset must have at least 3 rows")

    if strategy != "random":
        raise NotImplementedError("Only 'random' strategy is supported in v0.1")

    if groups is not None or time_col is not None:
        raise NotImplementedError("groups and time_col not supported in v0.1")

    rng = np.random.default_rng(seed)

    # Compute global split sizes (used for non-stratified case + metadata)
    train_size, val_size, test_size = _validate_and_normalize_sizes(n, ratios)

    # Decide whether to stratify
    y = df[target]
    nunique = int(y.nunique(dropna=False))
    categorical_like = (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
        or nunique <= min(50, max(2, int(0.1 * n)))
    )

    if stratify is None:
        do_stratify = categorical_like
    else:
        do_stratify = bool(stratify)
        if do_stratify and not categorical_like:
            raise ValueError(
                f"stratify=True but target '{target}' looks continuous (nunique={nunique}); "
                "set stratify=False for regression/continuous targets."
            )

    if not do_stratify:
        # Plain random split
        indices = np.arange(n, dtype=int)
        rng.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size :]
    else:
        # Stratified split: split indices within each class label then combine
        train_parts = []
        val_parts = []
        test_parts = []

        # groupby on the target values; preserves exact label sets present in y
        for label, idxs in y.groupby(y, dropna=False).groups.items():
            idxs = np.asarray(list(idxs), dtype=int)
            rng.shuffle(idxs)

            k = len(idxs)
            if k < 3:
                raise ValueError(
                    f"Cannot stratify into 3 splits: class {label!r} has only {k} samples."
                )

            t, v, te = _validate_and_normalize_sizes(k, ratios)

            train_parts.append(idxs[:t])
            val_parts.append(idxs[t : t + v])
            test_parts.append(idxs[t + v :])

        train_idx = np.concatenate(train_parts).astype(int, copy=False)
        val_idx = np.concatenate(val_parts).astype(int, copy=False)
        test_idx = np.concatenate(test_parts).astype(int, copy=False)

        # Shuffle within each split to remove label blocks (still deterministic)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        # Safety checks
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        if len(np.unique(all_idx)) != n:
            raise RuntimeError("Internal error: stratified split produced overlap or missing indices.")

    metadata = {
        "strategy": "random",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "stratified": bool(do_stratify),
        "target_nunique": nunique,
    }

    if do_stratify:
        metadata["target_counts_overall"] = {k: int(v) for k, v in y.value_counts(dropna=False).to_dict().items()}
        metadata["target_counts_train"] = {k: int(v) for k, v in y.iloc[train_idx].value_counts(dropna=False).to_dict().items()}
        metadata["target_counts_val"] = {k: int(v) for k, v in y.iloc[val_idx].value_counts(dropna=False).to_dict().items()}
        metadata["target_counts_test"] = {k: int(v) for k, v in y.iloc[test_idx].value_counts(dropna=False).to_dict().items()}

    return SplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata=metadata,
    )