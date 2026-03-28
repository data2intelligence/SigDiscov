# Custom Weight Matrix Generator

Generate spatial weight matrices from VST file headers using the same RBF kernel algorithm as the C program.

See [docs/weight-generator.md](docs/weight-generator.md) for full documentation.

## Quick Start

```bash
# Generate weight matrices (outputs: weights_dense.tsv, weights_sparse.tsv)
python3 make_custom_w.py -i expression.tsv -o weights -r 3 -s 0 -p 0

# Use with Moran's I
./morans_i_mkl -i expression.tsv -o results -w weights_dense.tsv --weight-format dense -b 1 -g 1

# Validate against built-in calculation
./morans_i_mkl -i expression.tsv -o builtin -r 3 -s 0 -p 0 -b 1 -g 1
diff builtin_all_pairs_moran_i_raw.tsv results_all_pairs_moran_i_raw.tsv
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input VST file | required |
| `-o, --output` | Output file prefix | vst_weights |
| `-r, --max-radius` | Maximum neighbor radius | 5 |
| `-s, --include-same-spot` | Include self-connections (0/1) | 0 |
| `-p, --platform` | 0=Visium, 1=Old ST, 2=Single Cell | 0 |
| `--sigma` | RBF kernel sigma | 100.0 |

## Dependencies

```bash
pip install -r requirements.txt
```
