import sys
import numpy as np
import scipy.sparse.linalg as spla

def load_matrix_from_txt(filename):
    """Loads a dense matrix from a text file."""
    return np.loadtxt(filename)

def load_vector_from_txt(filename):
    """Loads a dense vector from a text file."""
    vec = np.loadtxt(filename)
    return vec if vec.ndim == 1 else vec.flatten()

def run_cg(A, b, tol=1e-12, maxiter=None):
    print("Running Conjugate Gradient...")
    x, info = spla.cg(A, b, rtol=tol, maxiter=maxiter)

    if info == 0:
        print("CG converged successfully.")
    elif info > 0:
        print(f"CG stopped after {info} iterations (did not converge).")
    else:
        print("CG failed due to numerical error.")

    residual = np.linalg.norm(b - A @ x)
    print(f"Final residual norm: {residual:.2e}")
    return x

def main():
    A = load_matrix_from_txt(sys.argv[1])
    b = load_vector_from_txt(sys.argv[2])

    if A.shape[0] != b.shape[0]:
        raise ValueError("Matrix and vector dimensions do not match.")

    x = run_cg(A, b)

    np.savetxt("solution.txt", x, fmt="%.6f")
    print("Solution saved to solution.txt")

if __name__ == "__main__":
    main()
