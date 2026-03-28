# Weight Matrix Generator

Python tool that generates custom weight matrices using the same algorithm as the C program.

## Usage

```bash
python3 make_custom_w.py -i <input.tsv> -o <output_prefix> [OPTIONS]
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

## Output

Generates two files:
- `<prefix>_dense.tsv` -- full weight matrix with spot names
- `<prefix>_sparse.tsv` -- three-column sparse format

## Validation Workflow

```bash
# Generate custom weights
python3 make_custom_w.py -i data.tsv -o weights -r 3 -p 0

# Run with custom weights
./morans_i_mkl -i data.tsv -o test_custom -w weights_dense.tsv --weight-format dense -b 1 -g 1

# Run with built-in weights (same parameters)
./morans_i_mkl -i data.tsv -o test_builtin -p 0 -r 3 -b 1 -g 1

# Results should be identical
diff test_custom_all_pairs_moran_i_raw.tsv test_builtin_all_pairs_moran_i_raw.tsv
```

## Custom Weight Matrices in Python

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

spots = pd.read_csv('coordinates.tsv', sep='\t')
coords = spots[['x', 'y']].values

# RBF weights
distances = cdist(coords, coords)
sigma = 100.0
weights = np.exp(-(distances**2) / (2 * sigma**2))
np.fill_diagonal(weights, 0.0)
weights[weights < 1e-12] = 0.0

# Save dense
df = pd.DataFrame(weights, index=spots['spot_id'], columns=spots['spot_id'])
df.index.name = 'spot_id'
df.to_csv('weights_dense.tsv', sep='\t')
```

Alternative distance functions:
```python
# K-nearest neighbors
k = 6
weights = np.zeros_like(distances)
for i in range(len(distances)):
    nearest = np.argsort(distances[i])[1:k+1]
    weights[i, nearest] = 1.0

# Inverse distance
weights = 1.0 / (1.0 + distances)

# Distance threshold
weights = (distances <= 150.0).astype(float)
```
