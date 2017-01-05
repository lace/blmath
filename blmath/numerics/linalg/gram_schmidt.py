'''
Helper functions that implement Gram-Schmidt:

    - orthogonalize: (params: v)
    - orthonormalize: (params: v)

    Returns an M x N np.array of the vectors processed via the Gram-Schmidt algorithm.

Args:

    - v: an M x N np.array of vectors, where the
         *rows* are the vectors, not the columns.

Usage:

    from blmath.geometry.gram_schmidt import gram_schmidt

    # ...

    o = gram_schmidt.orthogonalize(v)

    # do stuff with o...
'''

import numpy as np

def orthogonalize(v):
    o = []

    for vec in v:
        w = vec.copy()

        proj_cmp = sum(np.array([proj(u, w) for u in o]))
        w = w - proj_cmp

        o.append(w)

    return np.array(o)

def orthonormalize(v, indices=None):
    if indices is None:
        indices = range(len(v))

    res = orthogonalize(v)

    for idx in indices:
        v = res[idx]
        v /= np.linalg.norm(v)

    return res

def proj(u, v):
    '''
    Projection of u onto v. Helper method for the Gram-Schmidt algorithm.
    '''
    return (np.dot(u, v) / np.dot(u, u)) * u

def main():
    v = np.array([[3, 1], [2, 2]], dtype=float)

    print orthogonalize(v)
    print orthonormalize(v)

    v = np.array([[1., 1., 1.], [2., 1., 0.], [5., 1., 3.]])

    print orthogonalize(v)
    print orthonormalize(v)


if __name__ == '__main__':
    main()
