# Mathematical Details

## Univariate Moran's I (Single-Gene)

```
I = (Z^T W Z) / S0
```

- **Z**: standardized (z-score) gene expression vector
- **W**: spatial weight matrix
- **S0**: sum of all weights in W

## Bivariate Moran's I (Pairwise)

```
I = (Z_1^T W Z_2) / S0
```

- **Z_1, Z_2**: standardized expression vectors for two genes across n spots

## Residual Moran's I

1. **Regression**: Fit cell type model per gene
   ```
   X = Z * B + E
   ```

2. **Residuals**: Remove cell type effects
   ```
   R = X - Z * B_hat
   ```

3. **Residual Moran's I**: Standard Moran's I on normalized residuals
   ```
   I_residual = (R_norm^T W R_norm) / S0
   ```

With optional ridge regularization (lambda > 0):
```
B_hat = (Z^T Z + lambda * I)^{-1} Z^T X
```

## Spatial Weight Matrix

Built-in RBF kernel:
```
W_ij = exp(-(d_ij^2) / (2 * sigma^2))
```

| Platform | Sigma | Notes |
|----------|-------|-------|
| Visium | 100 um | Hexagonal grid geometry |
| Old ST | 200 um | Square grid |
| Single-cell | inferred | From nearest-neighbor distances |

- W_ij = 0 beyond maximum radius threshold
- W_ii = 0 by default (configurable with `-s 1`)

## Permutation Testing

1. Calculate observed Moran's I
2. Shuffle gene expression (or residuals) across spots
3. Recalculate Moran's I for each permutation
4. Compute:
   - **Z-score**: (observed - mean_perm) / std_perm
   - **P-value**: (count(|perm| >= |observed|) + 1) / (n_perm + 1)
