def lchol(A):
    import copy
    import numpy as np
    import scipy.sparse as sp
    from blmath.numerics.linalg import cholmod # pylint: disable=no-name-in-module

    # the cholmod port needs 64b data, while scipy sparse is 32b
    A64 = copy.copy(A)
    A64.indices = A.indices.astype(np.int64)
    A64.indptr = A.indptr.astype(np.int64)

    [L_ind, L_iptr, L_data, L_nonpsd, q] = cholmod.lchol_c(A64)

    L = sp.csc_matrix((L_data, L_ind, L_iptr), shape=A64.shape)
    S = sp.csc_matrix((np.ones(len(q)), (q, range(len(q)))), shape=(max(q)+1, len(q)))

    return L, L_nonpsd, S
