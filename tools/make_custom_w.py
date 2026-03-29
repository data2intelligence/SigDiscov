#!/usr/bin/env python3
"""
Generate Custom Weight Matrix Directly from VST File Headers

Reads coordinates from column names (e.g., "0x50", "0x52") and generates
spatial weight matrices using the same RBF kernel algorithm as the C program.

Output: dense TSV and sparse TSV weight matrix files.
"""

import math
import re
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


WEIGHT_THRESHOLD = 1e-12
ZERO_STD_THRESHOLD = 1e-8

PLATFORM_PARAMS = {
    0: {"name": "Visium", "shift_y": 0.5 * math.sqrt(3.0), "dist_unit": 100.0},
    1: {"name": "Old ST", "shift_y": 0.5, "dist_unit": 200.0},
    2: {"name": "Single Cell", "shift_y": 1.0, "dist_unit": 100.0},
}


def read_vst_coordinates(vst_file):
    """Read spot coordinates from VST file header (ROWxCOL format)."""
    print(f"Reading coordinates from VST file: {vst_file}")

    with open(vst_file, 'r') as f:
        header_line = f.readline().strip()

    columns = header_line.split('\t')
    coordinate_pattern = re.compile(r'^(\d+)x(\d+)$')

    positions = []
    skipped = []

    for col_name in columns:
        col_name = col_name.strip()
        match = coordinate_pattern.match(col_name)
        if match:
            positions.append({
                'spot_id': col_name,
                'array_row': int(match.group(1)),
                'array_col': int(match.group(2))
            })
        else:
            skipped.append(col_name)

    if skipped:
        print(f"Skipped {len(skipped)} non-coordinate columns: {skipped}")

    if not positions:
        raise ValueError(
            "No coordinate columns found in VST file. "
            "Expected format: 0x50, 1x52, etc."
        )

    df = pd.DataFrame(positions)
    print(f"Found {len(positions)} spots "
          f"({df['array_row'].min()}x{df['array_col'].min()} to "
          f"{df['array_row'].max()}x{df['array_col'].max()})")

    return df


def generate_weight_matrix(vst_file, sigma=100.0, max_radius=5,
                           include_self=False, platform=0):
    """Generate weight matrix from VST file using platform-specific RBF kernel."""
    positions = read_vst_coordinates(vst_file)
    n_spots = len(positions)

    params = PLATFORM_PARAMS.get(platform, PLATFORM_PARAMS[0])
    shift_y = params["shift_y"]
    dist_unit = params["dist_unit"]

    print(f"\nGenerating weight matrix for {n_spots} spots")
    print(f"Platform: {params['name']}, sigma={sigma}, "
          f"max_radius={max_radius}, include_self={include_self}")

    rows_array = positions['array_row'].values
    cols_array = positions['array_col'].values

    row_indices, col_indices, weight_values = [], [], []
    sigma_cutoff = 3.0 * sigma
    two_sigma_sq = 2.0 * sigma ** 2

    for i in range(n_spots):
        if i % 500 == 0:
            print(f"  Processing spot {i + 1}/{n_spots}")

        row_diffs = np.abs(rows_array - rows_array[i])
        col_diffs = np.abs(cols_array - cols_array[i])

        # Filter by max_radius -- matches C code exactly
        valid_mask = (row_diffs < max_radius) & (col_diffs < 2 * max_radius)
        if not include_self:
            valid_mask[i] = False

        valid_j = np.where(valid_mask)[0]
        if len(valid_j) == 0:
            continue

        # Vectorized distance and weight computation
        x_dists = 0.5 * col_diffs[valid_j] * dist_unit
        y_dists = row_diffs[valid_j] * shift_y * dist_unit
        d_physical = np.sqrt(x_dists ** 2 + y_dists ** 2)

        within_cutoff = d_physical <= sigma_cutoff
        d_physical = d_physical[within_cutoff]
        valid_j = valid_j[within_cutoff]

        weights = np.exp(-(d_physical ** 2) / two_sigma_sq)
        above_threshold = weights > WEIGHT_THRESHOLD
        valid_j = valid_j[above_threshold]
        weights = weights[above_threshold]

        n_new = len(valid_j)
        if n_new > 0:
            row_indices.extend([i] * n_new)
            col_indices.extend(valid_j.tolist())
            weight_values.extend(weights.tolist())

    weight_matrix = coo_matrix(
        (weight_values, (row_indices, col_indices)),
        shape=(n_spots, n_spots)
    )

    print(f"Generated {len(weight_values)} non-zero weights")
    print(f"Matrix density: {len(weight_values) / (n_spots ** 2) * 100:.3f}%")
    print(f"Sum of weights (S0): {sum(weight_values):.6f}")

    return weight_matrix, positions['spot_id'].tolist()


def save_weight_matrices(weight_matrix, spot_ids, prefix="vst_weights"):
    """Save weight matrices in dense TSV and sparse TSV formats."""
    # Dense format
    dense_weights = weight_matrix.toarray()
    df_dense = pd.DataFrame(dense_weights, index=spot_ids, columns=spot_ids)
    df_dense.index.name = 'spot_id'
    dense_file = f"{prefix}_dense.tsv"
    df_dense.to_csv(dense_file, sep='\t')
    print(f"Dense matrix saved: {dense_file}")

    # Sparse format
    sparse_file = f"{prefix}_sparse.tsv"
    with open(sparse_file, 'w') as f:
        f.write("spot1\tspot2\tweight\n")
        for i, j, w in zip(weight_matrix.row, weight_matrix.col,
                           weight_matrix.data):
            f.write(f"{spot_ids[i]}\t{spot_ids[j]}\t{w:.12f}\n")
    print(f"Sparse matrix saved: {sparse_file}")

    return dense_file, sparse_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate custom weight matrix from VST file'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input VST file')
    parser.add_argument('-o', '--output', default='vst_weights',
                        help='Output file prefix (default: vst_weights)')
    parser.add_argument('-r', '--max-radius', type=int, default=5,
                        help='Maximum radius for neighbors (default: 5)')
    parser.add_argument('-s', '--include-same-spot', type=int,
                        choices=[0, 1], default=0,
                        help='Include self-connections: 0=No, 1=Yes (default: 0)')
    parser.add_argument('-p', '--platform', type=int,
                        choices=[0, 1, 2], default=0,
                        help='Platform: 0=Visium, 1=Old ST, 2=Single Cell (default: 0)')
    parser.add_argument('--sigma', type=float, default=100.0,
                        help='RBF kernel sigma (default: 100.0)')

    args = parser.parse_args()

    print("=== Weight Matrix Generator ===")
    print(f"Input: {args.input}")
    print(f"Output prefix: {args.output}")
    print(f"Parameters: platform={args.platform}, max_radius={args.max_radius}, "
          f"include_self={args.include_same_spot}, sigma={args.sigma}")
    print()

    try:
        weight_matrix, spot_ids = generate_weight_matrix(
            args.input,
            sigma=args.sigma,
            max_radius=args.max_radius,
            include_self=(args.include_same_spot == 1),
            platform=args.platform
        )

        print("\nSaving weight matrices...")
        dense_file, sparse_file = save_weight_matrices(
            weight_matrix, spot_ids, args.output
        )

        print("\n=== Usage ===")
        print(f"Dense:   ./build/morans_i_mkl -i {args.input} -o out "
              f"-w {dense_file} --weight-format dense -b 1 -g 1")
        print(f"Sparse:  ./build/morans_i_mkl -i {args.input} -o out "
              f"-w {sparse_file} --weight-format sparse_tsv -b 1 -g 1")
        print(f"Builtin: ./build/morans_i_mkl -i {args.input} -o out "
              f"-p {args.platform} -r {args.max_radius} "
              f"-s {args.include_same_spot} -b 1 -g 1")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
