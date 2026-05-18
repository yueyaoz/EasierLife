import pandas as pd
from pandas.testing import assert_series_equal
from typing import List, Optional
from pathlib import Path
import numpy as np

def normalize_path(path: Path) -> Path:

    return Path(str(path).replace('\\', '/'))


def _gather_folder_file_sets(
        folder: Path,
        glob_pattern: str,
        ignore_set: set,
        ignore_dirs: Optional[List[str]] = None,
) -> set:

    issues = []
    if not folder.is_dir():
        issues.append(f"missing folder1: {normalize_path(str(folder))}")
    if issues:
        raise AssertionError("folder missing:\n " + "\n ".join(issues))

    ignore_dirs_set = set(ignore_dirs or [])

    rel = {
        f.relative_to(folder).as_posix()
        for f in folder.rglob(glob_pattern)
        if f.is_file()
        and f.name not in ignore_set
        and not any(f.relative_to(folder).as_posix() == ign or f.relative_to(folder).as_posix().endswith("/" + ign) for ign in ignore_set)
        and not any(part in ignore_dirs_set for part in f.relative_to(folder).parts)
    }
    return rel


def compare_csv_folders(
        expected_path: Path,
        actual_path: Path,
        ignore_files: list[str] | None = None,
        ignore_dirs: list[str] | None = None,
        *,
        rtol: float = 5e-2,
        atol: float = 0.0,
        check_missing: bool = True,
        ignore_columns_by_file: dict[str, List[str]] | None = None,
        auto_detect_horizontal: bool = True,
) -> None:

    ignore_columns_by_file = ignore_columns_by_file or {}
    ignore_set = set(ignore_files or [])
    issues: List[str] = []

    rel1 = _gather_folder_file_sets(expected_path, "*.csv", ignore_set, ignore_dirs)
    rel2 = _gather_folder_file_sets(actual_path, "*.csv", ignore_set, ignore_dirs)

    # check for missing/extra files
    if check_missing:
        for rel in sorted(rel1 - rel2):
            issues.append(f"missing file in folder2: {normalize_path(str(actual_path / rel))}")
        for rel in sorted(rel2 - rel1):
            issues.append(f"extra file in folder2: {normalize_path(str(actual_path / rel))}")

    # Content diffs for common files
    for rel in sorted(rel1 & rel2):
        csv1_path = expected_path / rel
        csv2_path = actual_path / rel
        cols_to_ignore = None
        for key, cols in ignore_columns_by_file.items():
            if rel == key or rel.endswith("/" + key):
                cols_to_ignore = cols
                break
        issues.extend(_compare_csv(csv1_path, csv2_path, rtol=rtol, atol=atol, ignore_columns=cols_to_ignore, auto_detect_horizontal=auto_detect_horizontal))

    if issues:
        raise AssertionError("file differences in:\n- " + "\n- ".join(issues))


def transpose_kv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    transpose a horizontal table
    """
    result = df.set_index(df.columns[0]).T.reset_index(drop=True)
    result.columns.name = None
    return result


def _is_horizontal_table(df: pd.DataFrame) -> bool:
    """
    Detect if a DataFrame is a horizontal table.
    A horizontal table typically has:
    - Many columns relative to rows (> rows * 3)
    - First column is non-numeric (labels/identifiers)
    - Most other columns are numeric
    """
    if df.shape[0] == 0 or df.shape[1] < 2:
        return False
    
    rows, cols = df.shape
    
    # Heuristic 1: Must have significantly more columns than rows
    if cols <= rows * 3:
        return False
    
    # Heuristic 2: First column should be non-numeric (labels/identifiers)
    first_col_dtype = df.iloc[:, 0].dtype
    if pd.api.types.is_numeric_dtype(first_col_dtype):
        return False
    
    # Heuristic 3: Most other columns should be numeric
    numeric_cols = sum(1 for i in range(1, cols) if pd.api.types.is_numeric_dtype(df.iloc[:, i].dtype))
    if numeric_cols < (cols - 1) * 0.7:  # At least 70% of remaining columns are numeric
        return False
    
    return True


def _compare_csv(
    csv1_path: Path,
    csv2_path: Path,
    *,
    rtol: float = 5e-2,
    atol: float = 0.0,
    ignore_columns: Optional[List[str]] = None,
    auto_detect_horizontal: bool = True,
) -> List[str]:

    diffs: List[str] = []
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    except Exception as e:
        diffs.append(f"ERROR reading '{normalize_path(csv1_path)}' and '{normalize_path(csv2_path)}': {e}")
        return diffs

    # Auto-detect and transpose horizontal tables if enabled
    if auto_detect_horizontal:
        is_horizontal_1 = _is_horizontal_table(df1)
        is_horizontal_2 = _is_horizontal_table(df2)
        
        if is_horizontal_1:
            df1 = transpose_kv_df(df1)
        if is_horizontal_2:
            df2 = transpose_kv_df(df2)

    if ignore_columns:
        cols_to_drop = set(ignore_columns)
        df1 = df1.drop(columns=[c for c in cols_to_drop if c in df1.columns])
        df2 = df2.drop(columns=[c for c in cols_to_drop if c in df2.columns])

    df1 = df1[sorted(df1.columns)]
    df2 = df2[sorted(df2.columns)]

    # check column match
    if not df1.columns.equals(df2.columns):
        diffs.append(
            f"column mismatch: '{normalize_path(csv1_path)}' and '{normalize_path(csv2_path)}'\n"
            f"  cols1={list(df1.columns)}\n  cols2={list(df2.columns)}"
        )
        return diffs

    # return a customized error message if possible, otherwise fall back to the original error of assert_series_equal
    for col in df1.columns:
        try:
            assert_series_equal(df1[col], df2[col], rtol=rtol, atol=atol)
        except AssertionError as e:
            diffs.append(
                _assert_series_equal_with_more_information(
                    df1[col], df2[col], col, label=str(normalize_path(csv1_path)),
                    rtol=rtol, atol=atol, original_error=str(e),
                )
            )
    return diffs


def _assert_series_equal_with_more_information(
    s1: pd.Series,
    s2: pd.Series,
    col: str,
    label: str,
    rtol: float,
    atol: float,
    original_error: str,
) -> str:
    prefix = f"data mismatch: '{label}' column '{col}': {original_error.splitlines()[0]}"

    if len(s1) != len(s2):
        return f"{prefix}\n  row count differs: expected {len(s1)}, actual {len(s2)}"

    try:
        mask = ~np.isclose(s1, s2, rtol=rtol, atol=atol, equal_nan=True)
        failing = pd.DataFrame({"expected": s1[mask], "actual": s2[mask]})
        return f"{prefix}\n{failing.to_string()}"
    except TypeError:
        return prefix


def _compare_npy_files(
    npy1_path: Path,
    npy2_path: Path,
    *,
    atol: float = 0.0,
    rtol: float = 1e-4,
    equal_nan: bool = True,
    allow_pickle: bool = False,
) -> List[str]:
    diffs: List[str] = []
    try:
        a = np.load(npy1_path, allow_pickle=allow_pickle)
        b = np.load(npy2_path, allow_pickle=allow_pickle)
    except Exception as e:
        diffs.append(
            f"ERROR reading '{normalize_path(npy1_path)}' and '{normalize_path(npy2_path)}': {e}"
        )
        return diffs

    # dtype check
    if a.dtype != b.dtype:
        diffs.append(
            f"dtype mismatch: '{normalize_path(npy1_path)}' and '{normalize_path(npy2_path)}'\n"
            f"  dtype1={a.dtype}\n  dtype2={b.dtype}"
        )

    # shape check
    if a.shape != b.shape:
        diffs.append(
            f"shape mismatch: '{normalize_path(npy1_path)}' and '{normalize_path(npy2_path)}' "
            f"{a.shape} != {b.shape}"
        )

    # data check (only if dtype and shape match)
    if not diffs:
        try:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        except AssertionError as e:
            max_abs = float(np.nanmax(np.abs(a - b)))
            diffs.append(
                f"data mismatch: '{normalize_path(npy1_path)}' and '{normalize_path(npy2_path)}' "
                f"{str(e).splitlines()[0]} (max_abs_diff={max_abs})"
            )

    return diffs



def compare_npy_folder(
    folder1: Path,
    folder2: Path,
    *,
    atol: float = 0.0,
    rtol: float = 1e-4,
    equal_nan: bool = True,
    allow_pickle: bool = False,
    ignore_files: Optional[List[str]] = None,
) -> None:
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    ignore_set = set(ignore_files or [])

    rel1 = _gather_folder_file_sets(folder1, "*.npy", ignore_set)
    rel2 = _gather_folder_file_sets(folder2, "*.npy", ignore_set)
    missing_in_folder2 = sorted(rel1 - rel2)
    extra_in_folder2 = sorted(rel2 - rel1)

    if missing_in_folder2 or extra_in_folder2:
        def _preview(items, limit=20):
            items = [str(x) for x in items]
            return items[:limit] + (["..."] if len(items) > limit else [])

        raise AssertionError(
            "npy folder mismatch:\n"
            f"  folder1={normalize_path(folder1)}\n"
            f"  folder2={normalize_path(folder2)}\n"
            f"  missing_in_folder2={_preview(missing_in_folder2)}\n"
            f"  extra_in_folder2={_preview(extra_in_folder2)}"
        )

    issues: List[str] = []
    for rel in sorted(rel1 & rel2):
        file_issues = _compare_npy_files(
            folder1 / rel,
            folder2 / rel,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
            allow_pickle=allow_pickle,
        )
        for issue in file_issues:
            issues.append(f"Mismatch in '{rel}': {issue}")

    if issues:
        raise AssertionError("npy file differences:\n- " + "\n- ".join(issues))







