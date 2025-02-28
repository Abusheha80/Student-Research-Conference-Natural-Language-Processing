import pickle
import numpy as np

sparse_matrix_path = "sparse_matrix.pkl"

with open(sparse_matrix_path, 'rb') as f:
    sparse_matrix = pickle.load(f)

#matrix shape
num_rows, num_columns = sparse_matrix.shape
print(f"Shape of Sparse Matrix: {num_rows} rows, {num_columns} columns")

# Convert sparse matrix to dense (numpy array) format to see values
dense_matrix = sparse_matrix.toarray()

print("First 5 Rows of Sparse Matrix in Dense Form:")
print(dense_matrix[:5])

# Check sparsity level
num_nonzero = sparse_matrix.count_nonzero()
sparsity = 100 * (1 - (num_nonzero / (num_rows * num_columns)))
print(f"Sparsity of Matrix: {sparsity:.2f}%")
