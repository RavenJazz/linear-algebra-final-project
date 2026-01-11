import numpy as np

# --- Functions to get the ref and rref ---
def row_echelon(A):
    r, c = A.shape
    if r == 0 or c == 0: return A
    for i in range(len(A)):
        if np.abs(A[i,0]) > 1e-12: break
    else:
        B = row_echelon(A[:,1:])
        return np.hstack([A[:,:1], B])
    if i > 0:
        A[[0, i]] = A[[i, 0]]
    A[0] = A[0] / A[0,0]
    A[1:] -= A[0] * A[1:,0:1]
    B = row_echelon(A[1:,1:])
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

def ref_to_rref(R):
    R = R.copy().astype(float)
    rows, cols = R.shape
    pivots = []
    for i in range(rows):
        nz = np.where(np.abs(R[i]) > 1e-12)[0]
        if nz.size: pivots.append((i, nz[0]))
    for i, pj in reversed(pivots):
        R[i] = R[i] / R[i, pj]
        for u in range(0, i):
            R[u] -= R[u, pj] * R[i]
    return R, [p[1] for p in pivots] 

# --- Functions for the 4 Subspaces ---

def get_null_space(A):
    """ Finds basis for Null Space N(A) using free variables """
    R, pivot_cols = ref_to_rref(row_echelon(A.copy()))
    rows, cols = R.shape
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    
    null_basis = []
    for f in free_cols:
        vec = np.zeros(cols)
        vec[f] = 1  # Set free variable to 1
        for r_idx, p_col in enumerate(pivot_cols):
            # For each pivot row, pivot_var = -(sum of free_var * coefficient)
            vec[p_col] = -R[r_idx, f]
        null_basis.append(vec)
    
    return np.array(null_basis).T

def calculate_all_subspaces(A):
    # 1. Column Space C(A)
    # Rule: Original columns of A where pivots appear in RREF
    R, pivot_cols = ref_to_rref(row_echelon(A.copy()))
    col_space = A[:, pivot_cols]
    
    # 2. Row Space C(A^T)
    # Rule: The non-zero rows of the RREF matrix
    # (We transpose them because we usually represent basis as columns)
    row_space = R[:len(pivot_cols), :].T
    
    # 3. Null Space N(A)
    # Rule: Special solutions to Ax = 0
    null_space = get_null_space(A)
    
    # 4. Left Null Space N(A^T)
    # Rule: Null space of the transpose
    left_null_space = get_null_space(A.T)
    
    return col_space, row_space, null_space, left_null_space


# --- Execution ---
A = np.array([[1, 2, 3, 1],
              [1, 1, 2, 1],
              [1, 2, 3, 1]], dtype='float')

rref_A = ref_to_rref(row_echelon(A.copy()))[0]
c_s, r_s, n_s, ln_s = calculate_all_subspaces(A)

# --- Prints the values ---
print("--- RREF of A: ---\n", rref_A)
print("\n--- Column Space --- \n", c_s)
print("\n--- Row Space --- \n", r_s)
print("\n--- Null Space --- \n", n_s)
print("\n--- Left Null Space --- \n", ln_s)