# Usage Example

Generate weight matrix with same parameters as Moran's I:

```bash
# Generate custom weight matrix
python3 generate_weights.py -i expression.tsv -o custom_weights.tsv -r 3 -s 0 -p 0

# Test with Moran's I using same parameters
./morans_i_mkl -i expression.tsv -o builtin -r 3 -s 0 -p 0 -b 1 -g 1
./morans_i_mkl -i expression.tsv -o custom -w custom_weights.tsv --weight-format dense -b 1 -g 1

# Check if identical
diff builtin_all_pairs_moran_i_raw.tsv custom_all_pairs_moran_i_raw.tsv
```

Parameters:
- `-r 3`: max radius 3
- `-s 0`: no self-connections 
- `-p 0`: Visium platform
- `-i`: input file
- `-o`: output file
