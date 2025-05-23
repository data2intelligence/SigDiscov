#!/usr/bin/env python3
"""
Generate Custom Weight Matrix Directly from VST File Headers
Reads coordinates directly from column names like "0x50", "0x52", etc.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import math
import re
import argparse

def read_vst_coordinates(vst_file):
    """
    Read coordinates directly from VST file headers
    """
    print(f"Reading coordinates from VST file: {vst_file}")
    
    # Read just the header line
    with open(vst_file, 'r') as f:
        header_line = f.readline().strip()
    
    # Split by tabs and get column names
    columns = header_line.split('\t')
    
    # Extract coordinates using regex pattern for "rowxcol" format
    coordinate_pattern = r'^(\d+)x(\d+)$'
    
    positions = []
    skipped_columns = []
    
    print(f"Total columns found: {len(columns)}")
    print(f"Processing all {len(columns)} potential coordinate columns")
    
    for col_name in columns:  # Process ALL columns in header row
        col_name = col_name.strip()
        match = re.match(coordinate_pattern, col_name)
        if match:
            positions.append({
                'spot_id': col_name,
                'array_row': int(match.group(1)),
                'array_col': int(match.group(2))
            })
        else:
            skipped_columns.append(col_name)
            print(f"Skipped column: '{col_name}' (doesn't match pattern)")
    
    if skipped_columns:
        print(f"Total skipped columns: {len(skipped_columns)}")
        print(f"Skipped: {skipped_columns}")
    
    if not positions:
        raise ValueError("No coordinate columns found in VST file! Expected format: 0x50, 1x52, etc.")
    
    df = pd.DataFrame(positions)
    print(f"✓ Found {len(positions)} spots with coordinates")
    print(f"✓ Coordinate range: {df['array_row'].min()}x{df['array_col'].min()} to {df['array_row'].max()}x{df['array_col'].max()}")
    
    return df

def platform_distance_calculation(row1, col1, row2, col2, platform=0):
    """
    Calculate distance between two spots using platform-specific geometry
    """
    # Calculate grid coordinate differences
    row_shift = abs(row1 - row2)
    col_shift = abs(col1 - col2)
    
    if platform == 0:  # Visium
        shift_factor_y_grid = 0.5 * math.sqrt(3.0)  # Hexagonal geometry factor
        dist_unit_physical = 100.0  # μm between adjacent spots
    elif platform == 1:  # Old ST
        shift_factor_y_grid = 0.5
        dist_unit_physical = 200.0
    else:  # Single Cell (platform == 2)
        shift_factor_y_grid = 1.0
        dist_unit_physical = 100.0
    
    # Convert to physical distance
    x_dist_physical = 0.5 * col_shift * dist_unit_physical
    y_dist_physical = row_shift * shift_factor_y_grid * dist_unit_physical
    
    # Total physical distance
    d_physical_total = math.sqrt(x_dist_physical**2 + y_dist_physical**2)
    
    return d_physical_total

def decay_function(d_physical, sigma=100.0):
    """
    RBF decay function - exact replica of the C code decay() function
    """
    if d_physical < 0.0:
        d_physical = 0.0
    
    ZERO_STD_THRESHOLD = 1e-8
    if sigma <= ZERO_STD_THRESHOLD:
        return 1.0 if abs(d_physical) < ZERO_STD_THRESHOLD else 0.0
    
    # Cutoff beyond 3 sigma (matches C code)
    if d_physical > 3.0 * sigma:
        return 0.0
    
    # RBF kernel
    return math.exp(-(d_physical**2) / (2.0 * sigma**2))

def generate_weight_matrix_from_vst(vst_file, sigma=100.0, max_radius=5, include_self=False, platform=0):
    """
    Generate weight matrix directly from VST file (optimized version)
    """
    # Read coordinates from VST headers
    positions = read_vst_coordinates(vst_file)
    n_spots = len(positions)
    
    platform_names = {0: "Visium", 1: "Old ST", 2: "Single Cell"}
    print(f"\nGenerating weight matrix for {n_spots} spots")
    print(f"Parameters: platform={platform} ({platform_names.get(platform, 'Unknown')}), sigma={sigma}μm, max_radius={max_radius}, include_self={include_self}")
    
    # Convert to numpy arrays for speed
    rows_array = positions['array_row'].values
    cols_array = positions['array_col'].values
    
    # Platform-specific constants
    if platform == 0:  # Visium
        shift_factor_y_grid = 0.5 * math.sqrt(3.0)
        dist_unit_physical = 100.0
    elif platform == 1:  # Old ST
        shift_factor_y_grid = 0.5
        dist_unit_physical = 200.0
    else:  # Single Cell
        shift_factor_y_grid = 1.0
        dist_unit_physical = 100.0
    
    # Store results
    rows, cols, weights = [], [], []
    WEIGHT_THRESHOLD = 1e-12
    sigma_cutoff = 3.0 * sigma
    
    print("Processing distances...")
    for i in range(n_spots):
        if i % 500 == 0:
            print(f"  Processing spot {i+1}/{n_spots}")
        
        # Vectorized distance calculation for spot i vs all others
        row_diffs = np.abs(rows_array - rows_array[i])
        col_diffs = np.abs(cols_array - cols_array[i])
        
        # Filter by max_radius first - matches C code exactly
        # C code: row_shift_abs < max_radius AND col_shift_abs < 2*max_radius
        row_valid = row_diffs < max_radius
        col_valid = col_diffs < (2 * max_radius)
        valid_mask = row_valid & col_valid
        if not include_self:
            valid_mask[i] = False
        
        valid_indices = np.where(valid_mask)[0]
        
        for j in valid_indices:
            # Calculate physical distance
            x_dist = 0.5 * col_diffs[j] * dist_unit_physical
            y_dist = row_diffs[j] * shift_factor_y_grid * dist_unit_physical
            d_physical = math.sqrt(x_dist**2 + y_dist**2)
            
            # Apply RBF kernel with cutoff
            if d_physical <= sigma_cutoff:
                weight = math.exp(-(d_physical**2) / (2.0 * sigma**2))
                if weight > WEIGHT_THRESHOLD:
                    rows.append(i)
                    cols.append(j)
                    weights.append(weight)
    
    # Create sparse matrix
    weight_matrix = coo_matrix((weights, (rows, cols)), shape=(n_spots, n_spots))
    
    print(f"✓ Generated {len(weights)} non-zero weights")
    print(f"✓ Matrix density: {len(weights)/(n_spots**2)*100:.3f}%")
    print(f"✓ Sum of weights (S0): {sum(weights):.6f}")
    
    return weight_matrix, positions['spot_id'].tolist()

def save_weight_matrices(weight_matrix, spot_ids, prefix="vst_weights"):
    """
    Save weight matrices in formats supported by Moran's I code
    """
    # 1. Save as dense TSV format
    dense_weights = weight_matrix.toarray()
    df_dense = pd.DataFrame(dense_weights, index=spot_ids, columns=spot_ids)
    df_dense.index.name = 'spot_id'
    dense_filename = f"{prefix}_dense.tsv"
    df_dense.to_csv(dense_filename, sep='\t')
    print(f"✓ Dense matrix saved: {dense_filename}")
    
    # 2. Save as sparse TSV format (spot1, spot2, weight)
    sparse_filename = f"{prefix}_sparse.tsv"
    with open(sparse_filename, 'w') as f:
        f.write("spot1\tspot2\tweight\n")
        for i, j, w in zip(weight_matrix.row, weight_matrix.col, weight_matrix.data):
            f.write(f"{spot_ids[i]}\t{spot_ids[j]}\t{w:.12f}\n")
    print(f"✓ Sparse matrix saved: {sparse_filename}")
    
    return dense_filename, sparse_filename

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Generate custom weight matrix from VST file')
    parser.add_argument('-i', '--input', required=True, help='Input VST file')
    parser.add_argument('-o', '--output', default='vst_weights', help='Output file prefix (default: vst_weights)')
    parser.add_argument('-r', '--max-radius', type=int, default=5, help='Maximum radius for neighbors (default: 5)')
    parser.add_argument('-s', '--include-same-spot', type=int, choices=[0,1], default=0, help='Include self-connections: 0=No, 1=Yes (default: 0)')
    parser.add_argument('-p', '--platform', type=int, choices=[0,1,2], default=0, help='Platform: 0=Visium, 1=Old ST, 2=Single Cell (default: 0)')
    parser.add_argument('--sigma', type=float, default=100.0, help='RBF kernel sigma (default: 100.0)')
    
    args = parser.parse_args()
    
    print("=== Direct VST Weight Matrix Generator ===")
    print(f"Input file: {args.input}")
    print(f"Parameters: platform={args.platform}, max_radius={args.max_radius}, include_self={args.include_same_spot}, sigma={args.sigma}")
    print(f"Output prefix: {args.output}")
    print("")
    
    try:
        # Generate weight matrix
        weight_matrix, spot_ids = generate_weight_matrix_from_vst(
            args.input, 
            sigma=args.sigma, 
            max_radius=args.max_radius, 
            include_self=(args.include_same_spot == 1),
            platform=args.platform
        )
        
        # Save matrices
        print(f"\nSaving weight matrices...")
        dense_file, sparse_file = save_weight_matrices(weight_matrix, spot_ids, args.output)
        
        # Print usage instructions
        print(f"\n=== Usage Instructions ===")
        print(f"Test with dense format:")
        print(f"  ./morans_i_mkl -i {args.input} -o test_dense -w {dense_file} --weight-format dense -b 1 -g 1")
        print(f"\nTest with sparse format:")
        print(f"  ./morans_i_mkl -i {args.input} -o test_sparse -w {sparse_file} --weight-format sparse_tsv -b 1 -g 1")
        print(f"\nCompare with built-in calculation:")
        print(f"  ./morans_i_mkl -i {args.input} -o test_builtin -p {args.platform} -r {args.max_radius} -s {args.include_same_spot} -b 1 -g 1")
        
        print(f"\nTo validate results are identical:")
        print(f"  diff test_dense_all_pairs_moran_i_raw.tsv test_builtin_all_pairs_moran_i_raw.tsv")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())