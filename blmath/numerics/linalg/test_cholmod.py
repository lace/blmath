import unittest
from baiji.serialization import pickle
import numpy as np
import scipy.sparse as sp
from bltest import attr
from blmath.cache import vc
from blmath.numerics.linalg import lchol

class TestCholmod(unittest.TestCase):

    def test_random_cholmod(self):
        n_rows = 100
        A0 = 10*sp.rand(n_rows, n_rows, density=0.01, format='csc')
        A = A0*A0.transpose() + sp.eye(n_rows, n_rows)

        [L, L_nonpsd, S] = lchol.lchol(A)

        self.assertTrue(sum((abs(S.T.dot(A.dot(S))-L.dot(L.T))).data) < 1e-5)
        self.assertEqual(L_nonpsd, 0)

    # def test_memory_leak(self):
    #     n_rows = 3000
    #     A0 = 10*sp.rand(n_rows, n_rows, density=0.001, format='csc')
    #     A = A0*A0.transpose() + sp.eye(n_rows, n_rows)
    #     # mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #     for i in range(50):
    #         [chol_L, L_nonpsd, chol_S] = lchol.lchol(A)
    #         import gc
    #         gc.collect()
    #     # mem1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #     #print(mem1 - mem0)
    #     self.assertTrue(True)

    @attr('missing_assets')
    def test_cholmod(self):
        A, chol_L, _, cv = pickle.load(vc('/unittest/linalg/cholmod.pkl'))

        c_data = np.ones(len(cv))/len(cv)
        c_rows = cv.flatten()
        c_cols = (np.zeros(len(cv))).astype(np.int32)
        c = sp.csc_matrix((c_data, (c_rows, c_cols)), shape=(A.shape[0], 1))
        Ac = sp.hstack([A, c], format='csc')

        AAc = Ac.dot(Ac.T)

        [chol_L_comp, L_nonpsd, chol_S_comp] = lchol.lchol(AAc)

        right = chol_S_comp.T.dot(AAc.dot(chol_S_comp))
        left = chol_L_comp.dot(chol_L_comp.T)

        self.assertTrue(sum((abs(right-left)).data))  # it's a reordered LLt decomposition
        self.assertEqual(sp.triu(chol_L, k=1).nnz, 0) # it's lower triangular'
        self.assertEqual(L_nonpsd, 0)                 # the input is positive definite
        # self.assertTrue(sum((abs(chol_L - chol_L_comp)).data) < 1e-1)
        # self.assertTrue(sum((abs(chol_S - chol_S_comp)).data) < 1e-1)
