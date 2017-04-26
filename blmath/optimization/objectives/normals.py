# pylint: disable=arguments-differ, invalid-unary-operand-type
import numpy as np
import chumpy as ch
import scipy.sparse as sp
from blmath.numerics.matlab import row, col

try:
    from opendr.geometry import TriNormals, VertNormals, TriNormalsScaled, NormalizedNx3  # This lets us import bodylabs.geometry.normals.NormalizedNx3 etc. whether or not opendr is available. pylint: disable=unused-import
except ImportError:
    class NormalizedNx3(ch.Ch):
        dterms = 'v'
        ss = None  # TODO Explain what this is.
        s = None  # TODO Explain what this is.
        s_inv = None  # TODO Explain what this is.

        def on_changed(self, which):
            if 'v' in which:
                self.ss = np.sum(self.v.r.reshape(-1, 3)**2, axis=1)
                self.ss[self.ss == 0] = 1e-10
                self.s = np.sqrt(self.ss)
                self.s_inv = 1. / self.s

        def compute_r(self):
            return (self.v.r.reshape(-1, 3) / col(self.s)).reshape(self.v.r.shape)

        def compute_dr_wrt(self, wrt):
            if wrt is not self.v:
                return None

            v = self.v.r.reshape(-1, 3)
            blocks = -np.einsum('ij,ik->ijk', v, v) * (self.ss**(-3./2.)).reshape((-1, 1, 1))
            for i in range(3):
                blocks[:, i, i] += self.s_inv

            if True: # pylint: disable=using-constant-test
                data = blocks.ravel()
                indptr = np.arange(0, (self.v.r.size+1)*3, 3)
                indices = col(np.arange(0, self.v.r.size))
                indices = np.hstack([indices, indices, indices])
                indices = indices.reshape((-1, 3, 3))
                indices = indices.transpose((0, 2, 1)).ravel()
                result = sp.csc_matrix((data, indices, indptr), shape=(self.v.r.size, self.v.r.size))
                return result
            else:
                matvec = lambda x: np.einsum('ijk,ik->ij', blocks, x.reshape((blocks.shape[0], 3))).ravel()
                return sp.linalg.LinearOperator((self.v.r.size, self.v.r.size), matvec=matvec)


    class VertNormals(ch.Ch):
        """If normalized==True, normals are normalized; otherwise they'll be about as long as neighboring edges."""

        dterms = 'v'
        terms = 'f', 'normalized'
        term_order = 'v', 'f'

        def on_changed(self, which):
            # pylint: disable=access-member-before-definition, attribute-defined-outside-init

            if not hasattr(self, 'normalized'):
                self.normalized = True

            if hasattr(self, 'v') and hasattr(self, 'f'):
                if 'f' not in which and hasattr(self, 'faces_by_vertex') and self.faces_by_vertex.shape[0]/3 == self.v.shape[0]:
                    self.tns.v = self.v
                else: # change in f or in size of v. shouldn't happen often.
                    f = self.f

                    IS = f.ravel()
                    JS = np.array([range(f.shape[0])]*3).T.ravel()
                    data = np.ones(len(JS))

                    IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
                    JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
                    data = np.concatenate((data, data, data))

                    sz = self.v.size
                    self.faces_by_vertex = sp.csc_matrix((data, (IS, JS)), shape=(sz, f.size))

                    self.tns = ch.Ch(lambda v: CrossProduct(TriEdges(f, 1, 0, v), TriEdges(f, 2, 0, v)))
                    self.tns.v = self.v

                    if self.normalized:
                        tmp = ch.MatVecMult(self.faces_by_vertex, self.tns)
                        self.normals = NormalizedNx3(tmp)
                    else:
                        test = self.faces_by_vertex.dot(np.ones(self.faces_by_vertex.shape[1]))
                        faces_by_vertex = sp.diags([1. / test], [0]).dot(self.faces_by_vertex).tocsc()
                        normals = ch.MatVecMult(faces_by_vertex, self.tns).reshape((-1, 3))
                        normals = normals / (ch.sum(normals ** 2, axis=1) ** .25).reshape((-1, 1))
                        self.normals = normals

        def compute_r(self):
            return self.normals.r.reshape((-1, 3))

        def compute_dr_wrt(self, wrt):
            if wrt is self.v:
                return self.normals.dr_wrt(wrt)



    def TriNormals(v, f):
        return NormalizedNx3(TriNormalsScaled(v, f))

    def TriNormalsScaled(v, f):
        return CrossProduct(TriEdges(f, 1, 0, v), TriEdges(f, 2, 0, v))



    class TriEdges(ch.Ch):
        # TODO What are cplus and cminus?
        terms = 'f', 'cplus', 'cminus'
        dterms = 'v'

        def compute_r(self):
            cplus = self.cplus
            cminus = self.cminus
            return _edges_for(self.v.r, self.f, cplus, cminus)

        def compute_dr_wrt(self, wrt):
            if wrt is not self.v:
                return None

            cplus = self.cplus
            cminus = self.cminus
            vplus = self.f[:, cplus]
            vminus = self.f[:, cminus]
            vplus3 = row(np.hstack([col(vplus*3), col(vplus*3+1), col(vplus*3+2)]))
            vminus3 = row(np.hstack([col(vminus*3), col(vminus*3+1), col(vminus*3+2)]))

            IS = row(np.arange(0, vplus3.size))
            ones = np.ones(vplus3.size)
            shape = (self.f.size, self.v.r.size)
            dr_vplus, dr_vminus = [
                sp.csc_matrix((ones, np.vstack([IS, item])), shape=shape)
                for item in vplus3, vminus3  # FIXME change item to a DAMP
            ]
            return dr_vplus - dr_vminus

    def _edges_for(v, f, cplus, cminus):
        return (
            v.reshape(-1, 3)[f[:, cplus], :] -
            v.reshape(-1, 3)[f[:, cminus], :]).ravel()

    class CrossProduct(ch.Ch):
        terms = []
        dterms = 'a', 'b'
        indices = None  # TODO Explain what this is.
        a1 = a2 = a3 = None  # TODO Explain what this is.
        b1 = b2 = b3 = None  # TODO Explain what this is.
        indptr = None  # TODO Explain what this is.

        def on_changed(self, which):
            if 'a' in which:
                a = self.a.r.reshape((-1, 3))
                self.a1 = a[:, 0]
                self.a2 = a[:, 1]
                self.a3 = a[:, 2]
            if 'b' in which:
                b = self.b.r.reshape((-1, 3))
                self.b1 = b[:, 0]
                self.b2 = b[:, 1]
                self.b3 = b[:, 2]

        def compute_r(self):
            # TODO: maybe use cross directly? is it faster?
            # TODO: check fortran ordering?
            return _call_einsum_matvec(self.Ax, self.b.r)

        def compute_dr_wrt(self, obj):
            if obj not in (self.a, self.b):
                return None
            sz = self.a.r.size
            if self.indices is None or self.indices.size != sz*3:
                self.indptr = np.arange(0, (sz+1)*3, 3)
                idxs = col(np.arange(0, sz))
                idxs = np.hstack([idxs, idxs, idxs])
                idxs = idxs.reshape((-1, 3, 3))
                idxs = idxs.transpose((0, 2, 1)).ravel()
                self.indices = idxs
            if obj is self.a:
                # m = self.Bx
                # matvec = lambda x : _call_einsum_matvec(m, x)
                # matmat = lambda x : _call_einsum_matmat(m, x)
                # return sp.linalg.LinearOperator((self.a1.size*3, self.a1.size*3), matvec=matvec, matmat=matmat)
                data = self.Bx.ravel()
                result = sp.csc_matrix((data, self.indices, self.indptr), shape=(sz, sz))
                return -result
            elif obj is self.b:
                # m = self.Ax
                # matvec = lambda x : _call_einsum_matvec(m, x)
                # matmat = lambda x : _call_einsum_matmat(m, x)
                # return sp.linalg.LinearOperator((self.a1.size*3, self.a1.size*3), matvec=matvec, matmat=matmat)
                data = self.Ax.ravel()
                result = sp.csc_matrix((data, self.indices, self.indptr), shape=(sz, sz))
                return -result

        @ch.depends_on('a')
        def Ax(self):
            """Compute a stack of skew-symmetric matrices which can be multiplied
            by 'b' to get the cross product. See:
            http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
            """
            #  0         -self.a3   self.a2
            #  self.a3    0        -self.a1
            # -self.a2    self.a1   0
            m = np.zeros((len(self.a1), 3, 3))
            m[:, 0, 1] = -self.a3
            m[:, 0, 2] = +self.a2
            m[:, 1, 0] = +self.a3
            m[:, 1, 2] = -self.a1
            m[:, 2, 0] = -self.a2
            m[:, 2, 1] = +self.a1
            return m

        @ch.depends_on('b')
        def Bx(self):
            """Compute a stack of skew-symmetric matrices which can be multiplied
            by 'a' to get the cross product. See:
            http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
            """
            #  0         self.b3  -self.b2
            # -self.b3   0         self.b1
            #  self.b2  -self.b1   0


            m = np.zeros((len(self.b1), 3, 3))
            m[:, 0, 1] = +self.b3
            m[:, 0, 2] = -self.b2
            m[:, 1, 0] = -self.b3
            m[:, 1, 2] = +self.b1
            m[:, 2, 0] = +self.b2
            m[:, 2, 1] = -self.b1
            return m


    def _call_einsum_matvec(m, righthand):
        r = righthand.reshape(m.shape[0], 3)
        return np.einsum('ijk,ik->ij', m, r).ravel()

    def _call_einsum_matmat(m, righthand):
        r = righthand.reshape(m.shape[0], 3, -1)
        return np.einsum('ijk,ikm->ijm', m, r).reshape(-1, r.shape[2])
