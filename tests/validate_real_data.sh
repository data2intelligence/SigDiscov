#!/bin/bash
#SBATCH --job-name=sigdiscov_validate
#SBATCH --partition=norm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/vf/users/parks34/projects/0sigdiscov/pkg_dev/data2intelligence/SigDiscov/tests/output/validate_%j.log

set -euo pipefail

PROJECT_DIR="/vf/users/parks34/projects/0sigdiscov/pkg_dev/data2intelligence/SigDiscov"
cd "$PROJECT_DIR"

INPUT="/data/parks34/projects/0sigdiscov/archive/moran_i/datasets/visium/vst/1_vst.tsv"
EXPECTED="/data/parks34/projects/0sigdiscov/archive/moran_i/datasets/1_spatial_sig_vst_inhouse_s0_r3.tsv"
OUTPUT_DIR="${PROJECT_DIR}/tests/output"
OUTPUT_PREFIX="${OUTPUT_DIR}/validate_1_vst"

mkdir -p "$OUTPUT_DIR"

module load intel/2024.0.1.46

echo "=== Building ==="
make clean && make
echo ""

echo "=== Running Moran's I (Visium, -s 0 -r 3, all pairs) ==="
time ./morans_i_mkl \
    -i "$INPUT" \
    -o "$OUTPUT_PREFIX" \
    -p 0 -r 3 -s 0 \
    -b 1 -g 1 \
    -t 8

echo ""

ACTUAL="${OUTPUT_PREFIX}_all_pairs_moran_i_raw.tsv"

if [ ! -f "$ACTUAL" ]; then
    echo "FAIL: Output file not created: $ACTUAL"
    exit 1
fi

echo "=== Comparing output ==="
echo "Expected: $EXPECTED ($(wc -l < "$EXPECTED") lines)"
echo "Actual:   $ACTUAL ($(wc -l < "$ACTUAL") lines)"

expected_lines=$(wc -l < "$EXPECTED")
actual_lines=$(wc -l < "$ACTUAL")

if [ "$expected_lines" != "$actual_lines" ]; then
    echo "FAIL: Line count mismatch: expected=$expected_lines actual=$actual_lines"
    exit 1
fi

echo "Line counts match: $actual_lines"

python3 -c "
import sys
import numpy as np

tolerance = 1e-6
expected_file = '$EXPECTED'
actual_file = '$ACTUAL'

# Load as ragged text, parse each row as floats
print('Loading expected output...')
e_rows = []
with open(expected_file) as f:
    for line in f:
        e_rows.append(np.array(line.strip().split('\t'), dtype=np.float64))

print('Loading actual output...')
a_rows = []
with open(actual_file) as f:
    for line in f:
        a_rows.append(np.array(line.strip().split('\t'), dtype=np.float64))

if len(e_rows) != len(a_rows):
    print(f'FAIL: Row count mismatch: expected={len(e_rows)} actual={len(a_rows)}')
    sys.exit(1)

total_count = 0
max_diff = 0.0
diff_count = 0

for line_num, (e_vals, a_vals) in enumerate(zip(e_rows, a_rows), 1):
    if len(e_vals) != len(a_vals):
        print(f'FAIL: Column count mismatch at line {line_num}: expected={len(e_vals)} actual={len(a_vals)}')
        sys.exit(1)
    diffs = np.abs(e_vals - a_vals)
    row_max = diffs.max()
    if row_max > max_diff:
        max_diff = row_max
    bad = diffs > tolerance
    n_bad = bad.sum()
    total_count += len(e_vals)
    if n_bad > 0 and diff_count < 5:
        idxs = np.where(bad)[0]
        for idx in idxs[:5 - diff_count]:
            print(f'  Diff at line {line_num}, col {idx+1}: expected={e_vals[idx]:.10f} actual={a_vals[idx]:.10f} diff={diffs[idx]:.2e}')
    diff_count += n_bad

print()
print(f'Total values compared: {total_count}')
print(f'Max absolute difference: {max_diff:.2e}')
print(f'Values exceeding tolerance ({tolerance}): {diff_count}')

if diff_count == 0:
    print('PASS: Output matches expected within tolerance')
    sys.exit(0)
else:
    print(f'FAIL: {diff_count} values differ beyond tolerance')
    sys.exit(1)
"

echo ""
echo "=== Validation complete ==="
