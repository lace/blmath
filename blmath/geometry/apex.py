import numpy as np
from blmath.numerics import vx

def apex(points, axis):
    '''
    Find the most extreme point in the direction of the axis provided.

    axis: A vector, which is an 3x1 np.array.

    '''
    coords_on_axis = points.dot(axis)
    return points[np.argmax(coords_on_axis)]

def farthest(from_point, to_points):
    '''
    Find the farthest point among the inputs, to the given point.

    Return a tuple: farthest_point, index_of_farthest_point.
    '''
    absolute_distances = vx.magnitude(to_points - from_point)

    index_of_farthest_point = np.argmax(absolute_distances)
    farthest_point = to_points[index_of_farthest_point]

    return farthest_point, index_of_farthest_point
