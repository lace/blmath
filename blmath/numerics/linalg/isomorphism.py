import numpy as np


class DimensionalityError(ValueError):
    pass


class LinearDependenceError(ValueError):
    pass


def isomorphism(frame1, frame2):
    '''
    Takes two bases and returns their standard matrix representation.

    Args:

        frame1: N x N np.ndarray
        frame2: N x N np.ndarray

    Returns:
        The matrix representation of the corresponding operator
        in Euclidean coordinates.
    '''
    def _validate_frames(frame1, frame2):
        '''
        Checks for linear dependence and dimensionality errors.
        '''
        if len(frame1) != len(frame2):
            raise DimensionalityError('Bases must have the same length.')

        L_frame1_e = np.matrix(frame1).T
        L_frame2_e = np.matrix(frame2).T

        if len(L_frame1_e.shape) != 2:
            raise DimensionalityError('Extra dimensions: %s' % L_frame1_e)
        elif len(L_frame2_e.shape) != 2:
            raise DimensionalityError('Extra dimensions: %s' % L_frame2_e)
        elif L_frame1_e.shape[1] != L_frame2_e.shape[1]:
            raise DimensionalityError('Basis vectors must all have the same dimension.')
        elif np.linalg.det(L_frame1_e) == 0:
            raise LinearDependenceError('%s is linearly dependent.' % frame1)
        elif np.linalg.det(L_frame2_e) == 0:
            raise LinearDependenceError('%s is linearly dependent.' % frame2)

    _validate_frames(frame1, frame2)

    L_frame1_e = np.matrix(frame1).T
    L_frame2_e = np.matrix(frame2).T

    L_e_frame1 = L_frame1_e.I
    L_e_e = L_e_frame1 * L_frame2_e

    return L_e_e
