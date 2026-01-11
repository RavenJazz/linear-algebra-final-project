from pyscript import display
import numpy as np

def fundamental_subspaces(A, tol=1e-10):
    """
    Calculates the basis for the four fundamental subspaces of matrix A using SVD.
    Returns: Column Space, Null Space, Row Space, Left Null Space
    """
    m, n = A.shape
    
    # Perform Singular Value Decomposition
    # A = U * S * Vh
    U, S, Vh = np.linalg.svd(A)
    V = Vh.T.conj()

    # Determine the rank of the matrix based on a tolerance
    rank = np.sum(S > tol)

    # 1. Column Space C(A): First 'rank' columns of U
    col_space = U[:, :rank]

    # 2. Left Null Space N(A^T): Remaining columns of U
    left_null_space = U[:, rank:]

    # 3. Row Space C(A^T): First 'rank' columns of V (or rows of Vh)
    row_space = V[:, :rank]

    # 4. Null Space N(A): Remaining columns of V
    null_space = V[:, rank:]

    return col_space, null_space, row_space, left_null_space

# --- Execution ---
A = np.array([[1, 2, 3], 
              [2, 4, 6]])

col, null, row, l_null = fundamental_subspaces(A)

# --- Prints the values ---
print("Column Space Basis:\n", col)
print("\nNull Space Basis:\n", null)
print("\nRow Space Basis:\n", row)
print("\nLeft Null Space Basis:\n", l_null)