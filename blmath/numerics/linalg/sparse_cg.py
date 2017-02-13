def block_sparse_cg_solve(A, x):
    '''
    This function can be used by the optimize to solve the sparse matrix A
    (which will be J.T.dot(J), where J is the Jacobian of the objective
    function). The structure of A is such that it is mostly sparse with a few
    dense rows and columns (and also some columns with all zero). By
    explicitly spliting the matrix A into dense and sparse components, we can
    construct a basic preconditioner that makes linalg.cg much faster. (A
    preconditioner matrix for a matrix A is a matrix that approximates the
    inverse of A.)
    '''
    import numpy as np
    import scipy
    from blmath.numerics.linalg.static_block_sparse_matrix import StaticBlockSparseMatrix

    B = A.copy()
    B.data = (0.*B.data + 1.0)
    A_sparsity = np.array(np.bitwise_and((B.sum(axis=0) < 1000), (B.sum(axis=0) > 0))).flatten()
    A_block_sparse = StaticBlockSparseMatrix(A, A_sparsity, A_sparsity)
    non_zero_columns = np.array(((A_block_sparse.block_dd != 0).sum(axis=0) > 0)).flatten()
    A_inv_dense_corner = scipy.linalg.inv(A_block_sparse.block_dd[non_zero_columns, :][:, non_zero_columns])
    A_sparse_corner_solver = scipy.sparse.linalg.factorized(A_block_sparse.sparse_block.tocsc())
    def A_inv_mult(v):
        ans_sparse = A_sparse_corner_solver(v[A_block_sparse.sparse_row_indices].flatten())
        ans_dense = A_inv_dense_corner.dot(v[A_block_sparse.dense_row_indices[non_zero_columns]])
        ans = np.zeros(len(A_block_sparse.sparse_row_indices) + len(A_block_sparse.dense_row_indices))
        ans[A_block_sparse.sparse_row_indices] = ans_sparse
        ans[A_block_sparse.dense_row_indices[non_zero_columns]] = ans_dense
        return ans

    precon = scipy.sparse.linalg.LinearOperator(A.shape, matvec=A_inv_mult)
    return scipy.sparse.linalg.cg(A, x, M=precon, tol=1e-5)[0]
