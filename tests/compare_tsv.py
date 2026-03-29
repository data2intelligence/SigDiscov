#!/usr/bin/env python3
"""Compare two lower-triangular TSV files within a floating-point tolerance.

Usage:
    python3 tests/compare_tsv.py <expected> <actual> [tolerance]

Exit codes:
    0  -- files match within tolerance
    1  -- files differ beyond tolerance or structural mismatch
"""

import sys
import numpy as np


def compare(expected_file, actual_file, tolerance=1e-6):
    print(f"Loading expected: {expected_file}")
    e_rows = []
    with open(expected_file) as f:
        for line in f:
            e_rows.append(np.array(line.strip().split('\t'), dtype=np.float64))

    print(f"Loading actual:   {actual_file}")
    a_rows = []
    with open(actual_file) as f:
        for line in f:
            a_rows.append(np.array(line.strip().split('\t'), dtype=np.float64))

    if len(e_rows) != len(a_rows):
        print(f"FAIL: Row count mismatch: expected={len(e_rows)} actual={len(a_rows)}")
        return 1

    total_count = 0
    max_diff = 0.0
    diff_count = 0

    for line_num, (e_vals, a_vals) in enumerate(zip(e_rows, a_rows), 1):
        if len(e_vals) != len(a_vals):
            print(f"FAIL: Column count mismatch at line {line_num}: "
                  f"expected={len(e_vals)} actual={len(a_vals)}")
            return 1
        diffs = np.abs(e_vals - a_vals)
        row_max = float(diffs.max())
        if row_max > max_diff:
            max_diff = row_max
        bad = diffs > tolerance
        n_bad = int(bad.sum())
        total_count += len(e_vals)
        if n_bad > 0 and diff_count < 5:
            idxs = np.where(bad)[0]
            for idx in idxs[:5 - diff_count]:
                print(f"  Diff at line {line_num}, col {idx+1}: "
                      f"expected={e_vals[idx]:.10f} actual={a_vals[idx]:.10f} "
                      f"diff={diffs[idx]:.2e}")
        diff_count += n_bad

    print()
    print(f"Total values compared: {total_count}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Values exceeding tolerance ({tolerance}): {diff_count}")

    if diff_count == 0:
        print("PASS: Output matches expected within tolerance")
        return 0
    else:
        print(f"FAIL: {diff_count} values differ beyond tolerance")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <expected> <actual> [tolerance]")
        sys.exit(2)

    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    sys.exit(compare(sys.argv[1], sys.argv[2], tol))
