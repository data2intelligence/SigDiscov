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

tolerance = 1e-6
expected_file = '$EXPECTED'
actual_file = '$ACTUAL'

max_diff = 0.0
diff_count = 0
total_count = 0

with open(expected_file) as ef, open(actual_file) as af:
    for line_num, (e_line, a_line) in enumerate(zip(ef, af), 1):
        e_vals = e_line.strip().split('\t')
        a_vals = a_line.strip().split('\t')

        if len(e_vals) != len(a_vals):
            print(f'FAIL: Column count mismatch at line {line_num}: expected={len(e_vals)} actual={len(a_vals)}')
            sys.exit(1)

        for col, (ev, av) in enumerate(zip(e_vals, a_vals)):
            total_count += 1
            try:
                e_float = float(ev)
                a_float = float(av)
                diff = abs(e_float - a_float)
                if diff > max_diff:
                    max_diff = diff
                if diff > tolerance:
                    diff_count += 1
                    if diff_count <= 5:
                        print(f'  Diff at line {line_num}, col {col+1}: expected={e_float:.10f} actual={a_float:.10f} diff={diff:.2e}')
            except ValueError:
                if ev.strip() != av.strip():
                    diff_count += 1
                    if diff_count <= 5:
                        print(f'  String mismatch at line {line_num}, col {col+1}: expected=\"{ev}\" actual=\"{av}\"')

print(f'')
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
