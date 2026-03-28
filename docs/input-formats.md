# Input Formats

All files accept TSV or CSV with automatic format detection.

## Expression Data

Genes as rows, spots/cells as columns. First cell in header can be empty or a label.

**Visium/ST** -- header contains coordinates as `ROWxCOL`:
```
Gene    1x1   1x2   1x3   2x1   2x2
Gene1   0.5   1.2   0.8   0.3   1.5
Gene2   2.1   1.8   0.9   1.4   0.6
```

**Single-cell** -- header contains cell IDs matching the coordinate file:
```
Gene    cell_1  cell_2  cell_3
Gene1   0.5     1.2     0.8
Gene2   2.1     1.8     0.9
```

## Coordinate File (Single-Cell)

Required when `-p 2`. Columns for cell ID, X, and Y coordinates:
```
cell_ID,sdimx,sdimy
cell_1,10.5,20.3
cell_2,15.8,18.7
cell_3,12.1,25.4
```

Column names are configurable via `--id-col`, `--x-col`, `--y-col`.

## Cell Type Data

### Deconvolution Format (`--celltype-format deconv`)

Cell type proportions per spot. Either orientation:

**Spots as rows:**
```
spot_id,Tcell,Bcell,Macrophage,Epithelial
spot_1,0.3,0.1,0.2,0.4
spot_2,0.4,0.0,0.3,0.3
```

**Cell types as rows:**
```
cell_type,1x1,1x2,2x1,2x2
Tcell,0.3,0.4,0.2,0.1
Bcell,0.1,0.0,0.2,0.3
```

### Single-Cell Format (`--celltype-format sc`)

Individual cell annotations with coordinates:
```
cell_ID,cellType,sdimx,sdimy
cell_1,Tcell,10.5,20.3
cell_2,Macrophage,15.8,18.7
cell_3,Epithelial,12.1,25.4
```

## Custom Weight Matrices

### Dense TSV
Full matrix with spot names as headers and row labels:
```
spot_id 1x1 1x2 2x1 2x2
1x1 0.0 0.8 0.6 0.3
1x2 0.8 0.0 0.3 0.6
2x1 0.6 0.3 0.0 0.8
2x2 0.3 0.6 0.8 0.0
```

### Sparse COO
Coordinate format (row, column, weight):
```
row_spot  col_spot  weight
1x1 1x2 0.8
1x1 2x1 0.6
```

### Sparse TSV
Three-column format (spot1, spot2, weight):
```
spot1 spot2 weight
spot_A  spot_B  0.8
spot_A  spot_C  0.6
```

Format is auto-detected, or specify with `--weight-format`.
