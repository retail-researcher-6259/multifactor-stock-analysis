"""
scan_nan.py — Scan ranked-list CSV files for NaN / empty values.

Usage:
    python scan_nan.py <directory>

Example:
    python scan_nan.py output/Ranked_Lists/Crisis_Bear/oil
"""

import sys
import os
import pandas as pd


def scan_directory(directory: str) -> None:
    csv_files = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".csv")
    )

    if not csv_files:
        print(f"No CSV files found in: {directory}")
        return

    total_nan_cells = 0
    files_with_nan = 0

    print(f"Scanning {len(csv_files)} files in: {directory}\n")
    print("=" * 70)

    for filename in csv_files:
        filepath = os.path.join(directory, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"[ERROR] Could not read {filename}: {e}")
            continue

        nan_mask = df.isnull()
        nan_count = nan_mask.values.sum()

        if nan_count == 0:
            continue

        files_with_nan += 1
        total_nan_cells += nan_count
        print(f"File: {filename}  ({nan_count} NaN cell(s))")

        # Show per-column NaN counts
        col_nans = nan_mask.sum()
        col_nans = col_nans[col_nans > 0]
        for col, count in col_nans.items():
            print(f"  Column '{col}': {count} NaN(s)")

        # Show the affected rows (Ticker + columns that have NaN)
        nan_rows = df[nan_mask.any(axis=1)]
        affected_cols = col_nans.index.tolist()
        display_cols = (["Ticker"] if "Ticker" in df.columns else []) + affected_cols
        print(nan_rows[display_cols].to_string(index=True))
        print()

    print("=" * 70)
    if files_with_nan == 0:
        print("No NaN values found in any file.")
    else:
        print(
            f"Summary: {files_with_nan} file(s) with NaN values, "
            f"{total_nan_cells} total NaN cell(s)."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the oil directory when run without arguments
        default_dir = os.path.join(
            "output", "Ranked_Lists", "Crisis_Bear", "oil"
        )
        target = default_dir
    else:
        target = sys.argv[1]

    if not os.path.isdir(target):
        print(f"Directory not found: {target}")
        sys.exit(1)

    scan_directory(target)
