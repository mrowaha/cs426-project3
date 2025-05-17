import numpy as np
import sys
import math

def load_matrix(filename):
    try:
        return np.loadtxt(filename)
    except Exception as e:
        print(f"Error loading matrix from {filename}: {e}")
        sys.exit(1)

def extract_block(matrix, alpha, beta, p):
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    sqrt_p = int(math.isqrt(p))
    if sqrt_p * sqrt_p != p:
        raise ValueError("p must be a perfect square.")

    if n % sqrt_p != 0:
        raise ValueError("Matrix size n must be divisible by sqrt(p).")

    block_size = n // sqrt_p

    row_start = alpha * block_size
    row_end = (alpha + 1) * block_size
    col_start = beta * block_size
    col_end = (beta + 1) * block_size

    block = matrix[row_start:row_end, col_start:col_end]
    return block

def compare_blocks(extracted, loaded_block):
    if extracted.shape != loaded_block.shape:
        print(f"Shape mismatch: extracted {extracted.shape}, loaded {loaded_block.shape}")
        return False
    return np.allclose(extracted, loaded_block, atol=1e-6)

def main():
    full_matrix_file = "./example_input/example_m.txt"         # Path to full n x n matrix
    block_file = "A(0,0).txt"            # Path to block to validate
    alpha = 0                                # Row coordinate of processor
    beta = 0                                 # Column coordinate of processor
    p = 16                                  # Total number of processors (must be a perfect square)

    full_matrix = load_matrix(full_matrix_file)
    print(f"Loaded full matrix of shape {full_matrix.shape}")

    block_to_validate = load_matrix(block_file)
    print(f"Loaded block to validate of shape {block_to_validate.shape}")

    extracted_block = extract_block(full_matrix, alpha, beta, p)

    if compare_blocks(extracted_block, block_to_validate):
        print("✅ Block is valid!")
    else:
        print("❌ Block is invalid.")

if __name__ == "__main__":
    main()
