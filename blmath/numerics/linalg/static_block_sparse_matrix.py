import numpy as np
import scipy as sp

def mask_sparse_matrix_by_rows(M, row_mask):
    M_masked = sp.sparse.coo_matrix(M.shape, dtype=M.dtype)
    if np.sum(row_mask):
        M_coo = M.tocoo()
        entries_to_keep = row_mask[M_coo.row]
        M_masked = sp.sparse.coo_matrix((M_coo.data[entries_to_keep], (M_coo.row[entries_to_keep], M_coo.col[entries_to_keep])), shape=M.shape, dtype=M.dtype)
    return M_masked

def mask_sparse_matrix_by_columns(M, column_mask):
    M_masked = sp.sparse.coo_matrix(M.shape, dtype=M.dtype)
    if np.sum(column_mask):
        M_coo = M.tocoo()
        entries_to_keep = column_mask[M_coo.col]
        M_masked = sp.sparse.coo_matrix((M_coo.data[entries_to_keep], (M_coo.row[entries_to_keep], M_coo.col[entries_to_keep])), shape=M.shape, dtype=M.dtype)
    return M_masked

def mask_sparse_matrix(M, row_mask, column_mask):
    if not np.sum(column_mask):
        return mask_sparse_matrix_by_rows(M, row_mask)
    if not np.sum(row_mask):
        return mask_sparse_matrix_by_columns(M, column_mask)
    M_coo = M.tocoo()
    entries_to_keep = column_mask[M_coo.col]*row_mask[M_coo.row]
    return sp.sparse.coo_matrix((M_coo.data[entries_to_keep], (M_coo.row[entries_to_keep], M_coo.col[entries_to_keep])), shape=M.shape, dtype=M.dtype)

class StaticBlockSparseMatrix(object):

    def __init__(self, M, row_sparsity=None, column_sparsity=None, sparsity_threshold=0.05):
        # pylint: disable=len-as-condition
        self.M = M
        self.num_rows, self.num_columns = M.shape
        self.sparsity_threshold = sparsity_threshold*np.max(M.shape)

        self.M_csr = sp.sparse.csr_matrix(M)
        if row_sparsity is None:
            self.elements_per_row = np.array([self.M_csr.indptr[i + 1] - self.M_csr.indptr[i] for i in range(0, len(self.M_csr.indptr) - 1)])
            row_sparsity = self.elements_per_row < self.sparsity_threshold


        if column_sparsity is None:
            self.M_csc = sp.sparse.csc_matrix(M)
            self.elements_per_column = np.array([self.M_csc.indptr[i + 1] - self.M_csc.indptr[i] for i in range(0, len(self.M_csc.indptr) - 1)])
            column_sparsity = self.elements_per_column < self.sparsity_threshold

        self.r_s = row_sparsity if len(row_sparsity) else np.array([True]*self.M.shape[0])
        self.r_d = np.bitwise_not(self.r_s)
        self.ri_s = self.r_s.nonzero()[0]
        self.ri_d = self.r_d.nonzero()[0]

        self.c_s = column_sparsity if len(column_sparsity) else np.array([True]*self.M.shape[1])
        self.c_d = np.bitwise_not(self.c_s)
        self.ci_s = self.c_s.nonzero()[0]
        self.ci_d = self.c_d.nonzero()[0]

        M_coo = sp.sparse.coo_matrix(M)
        # sparse blocks s, and ss are created to be the size of the entire matrix, M. Dense blocks, however, are just the size of the subblocks.

        self.block_s = mask_sparse_matrix_by_rows(M_coo, self.row_sparsity)
        self.block_ss = mask_sparse_matrix_by_columns(self.block_s, self.column_sparsity).tocsr()
        self.block_ss_csc = self.block_ss.tocsc()
        self.block_sd = mask_sparse_matrix_by_columns(self.block_s, self.column_density).tocsr()[:, self.dense_column_indices].todense() if self.num_dense_columns else np.zeros((self.num_sparse_rows, self.num_dense_columns))
        self.block_s = self.block_s.tocsr()
        self.block_s_csc = self.block_s.tocsc()

        self.block_d_sparse = mask_sparse_matrix_by_rows(M_coo, self.row_density).tocsr()
        self.block_d = self.block_d_sparse[self.dense_row_indices, :].todense()
        self.block_ds = self.block_d[:, self.sparse_column_indices]
        self.block_dd = self.block_d[:, self.dense_column_indices]

        self.sparse_block = self.block_ss_csc[:, self.sparse_column_indices].tocsr()[self.sparse_row_indices, :]

    @property
    def shape(self):
        return (self.num_rows, self.num_columns)

    @property
    def row_sparsity(self):
        return self.r_s
    @property
    def row_density(self):
        return self.r_d
    @property
    def sparse_row_indices(self):
        return self.ri_s
    @property
    def dense_row_indices(self):
        return self.ri_d
    @property
    def num_sparse_rows(self):
        return len(self.ri_s)
    @property
    def num_dense_rows(self):
        return len(self.ri_d)

    @property
    def column_sparsity(self):
        return self.c_s
    @property
    def column_density(self):
        return self.c_d
    @property
    def sparse_column_indices(self):
        return self.ci_s
    @property
    def dense_column_indices(self):
        return self.ci_d
    @property
    def num_sparse_columns(self):
        return len(self.ci_s)
    @property
    def num_dense_columns(self):
        return len(self.ci_d)

    def MMT(self):
        if self.num_dense_rows and not self.num_dense_columns:
            ans = self.block_s.dot(self.block_s.T).tocoo()
            ans_sd = self.block_s.dot(self.block_d.T)
            ans_d = ans_sd.T.copy()
            ans_d[:, self.dense_row_indices] = self.block_d.dot(self.block_d.T)

            ans_sd = sp.sparse.coo_matrix(ans_sd)
            ans_sd.col = self.dense_row_indices[ans_sd.col]

            ans_d = sp.sparse.coo_matrix(ans_d)
            ans_d.row = self.dense_row_indices[ans_d.row]

            ans = sp.sparse.coo_matrix((np.hstack([ans.data, ans_d.data, ans_sd.data]), (np.hstack([ans.row, ans_d.row, ans_sd.row]), np.hstack([ans.col, ans_d.col, ans_sd.col]))), shape=ans.shape, dtype=ans.dtype)
            return ans.tocsr()
        else:
            return self.M.dot(self.M.T)

    def MTM(self):
        return self.M.T.dot(self.M)

    def matvec(self, v):
        b = self.block_ss.dot(v)
        b[self.r_s] += self.block_sd.dot(v[self.c_d])
        b[self.r_d] = self.block_ds.dot(v[self.c_s]) + self.block_dd.dot(v[self.c_d])
        return b

    def matmat(self, m):
        return self.M.dot(m)

    @property
    def linOp(self):
        return sp.sparse.linalg.LinearOperator(self, shape=self.shape, matvec=self.matvec, matmat=self.matmat) # FIXME pylint: disable=redundant-keyword-arg

    def conjGradSolve(self, b, x0=None, tol=1.0e-9, maxiter=None):
        n = len(b)
        x = x0 if x0 else np.zeros(b)

        r = b - self.matvec(x)
        s = r.copy()

        num_iterations = maxiter if maxiter else n
        i = None  # For pylint.
        for i in range(num_iterations):
            u = self.matvec(s)
            alpha = np.dot(s, r)/np.dot(s, u)
            x = x + alpha*s
            r = b - self.matvec(x)
            if np.sqrt(np.dot(r, r)) < tol:
                break
            else:
                beta = -np.dot(r, u)/np.dot(s, u)
                s = r + beta*s
        return x, i
