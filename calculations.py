import numpy as np

def row_echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

def ref_to_rref(R):
    R = R.copy().astype(float)
    rows, cols = R.shape
    # find pivots: pivot in row i is first non-zero col
    pivots = []
    for i in range(rows):
        nz = np.where(np.abs(R[i]) > 1e-12)[0]
        if nz.size:
            pivots.append((i, nz[0]))
    # back-substitute
    for i, pj in reversed(pivots):
        R[i] = R[i] / R[i, pj]
        for u in range(0, i):
            R[u] -= R[u, pj] * R[i]
    return R


def row_space(A, Acopy):
    R = ref_to_rref(row_echelon(A))
    rows, cols = R.shape
    basis = []
    checker = True
    for i in range(cols):
        for j in range(rows):
            if R[j, i] > 1:
                checker = False
                break
        if checker == True:
            basis.append(Acopy[:,i])
        checker = True
    basis = np.array(basis).transpose()
    return basis
            

A = np.array([[4, 7, 3, 8],
              [8, 3, 8, 7],
              [2, 9, 5, 3]], dtype='float')
Acopy = A.copy()


print(A)
B = ref_to_rref(row_echelon(A))
print(B)
C = row_space(A, Acopy)
print(C)

