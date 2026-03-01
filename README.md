# splitsmith

ML dataset splitting with built-in leakage detection.

splitsmith splits your data into train/val/test sets (or k-fold CV) and automatically audits for common data leakage patterns — index overlap, duplicate rows across splits, group leakage, and time leakage.

## Install

```bash
pip install git+https://github.com/Amorfati123/splitsmith-v0.1.git
```

Or clone and install locally:

```bash
git clone https://github.com/Amorfati123/splitsmith-v0.1.git
cd splitsmith-v0.1
pip install -e .
```

## Quick Start

### Split and Audit

```python
import pandas as pd
from splitsmith import split, audit

df = pd.read_csv("data.csv")

# Random stratified split
result = split(df, target="label", strategy="random", seed=42)
print(result.summary())

# Audit for leakage
report = audit(df, result, "label")
print(report)        # LeakageReport(0 error, 0 warn, 2 info)
print(report.ok)     # True
```

### Split Strategies

```python
# Group-aware split (no user appears in multiple splits)
result = split(df, target="label", strategy="group", groups="user_id")

# Chronological split (train < val < test)
result = split(df, target="label", strategy="time", time_col="timestamp")

# Group + time (groups sorted by latest timestamp)
result = split(df, target="label", strategy="group_time",
               groups="user_id", time_col="timestamp")
```

### K-Fold Cross-Validation

```python
from splitsmith import k_fold, audit_cv, audit_cv_summary

cv = k_fold(df, target="label", k=5, strategy="stratified")

# Audit every fold
reports = audit_cv(df, cv, "label")
summary = audit_cv_summary(reports)
print(summary["all_ok"])  # True
```

**CV strategies:** `basic`, `stratified`, `group`, `time` (forward-chaining), `group_time`

### Export

```python
from splitsmith import report_to_json, report_to_html

# JSON (for MLflow, W&B, etc.)
report_to_json(report, path="audit_report.json")

# HTML (for human review)
report_to_html(report, path="audit_report.html")
```

### CLI

Run audits directly from the terminal or CI pipelines:

```bash
# Single split audit (exit code 0=pass, 1=leakage, 2=error)
python -m splitsmith audit data.csv --target label --strategy group --groups user_id

# K-fold CV audit
python -m splitsmith audit-cv data.csv --target label --k 5 --strategy stratified

# Save reports
python -m splitsmith audit data.csv --target label --output report.json --html report.html
```

## Leakage Checks

| Check | Severity | What it catches |
|---|---|---|
| Index overlap | error | Same row index in multiple splits |
| Cross-split duplicates | error | Identical rows in different splits |
| Within-split duplicates | warn | Identical rows in the same split |
| Group leakage | error | Same entity (user, patient, etc.) in multiple splits |
| Time leakage | error/warn | Train data from after val/test period |

## API Reference

### Splitting
- `split(df, target, strategy, groups, time_col, ratios, seed, stratify)` → `SplitResult`
- `k_fold(df, target, k, strategy, groups, time_col, seed)` → `CVResult`

### Auditing
- `audit(df, split_result, target, groups, time_col)` → `LeakageReport`
- `audit_cv(df, cv_result, target, groups, time_col)` → `List[LeakageReport]`
- `audit_cv_summary(reports)` → `dict`

### Export
- `split_to_json(split_result, path)` → `str`
- `report_to_json(report, path)` → `str`
- `cv_to_json(cv_result, path)` → `str`
- `audit_cv_to_json(reports, path)` → `str`
- `report_to_html(report, title, path)` → `str`
- `audit_cv_to_html(reports, title, path)` → `str`

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT
