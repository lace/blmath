import numpy as np

def partition(v, partition_size=5):
    '''

    params:
        v:
            V x N np.array of points in N-space

        partition_size:
            how many partitions intervals for each segment?

    Fill in the line segments determined by v with equally
    spaced points - the space for each segment is determined
    by the length of the segment and the supplied partition size.

    '''
    src = np.arange(len(v) - 1)
    dst = src + 1

    diffs = v[dst] - v[src]

    sqdis = np.square(diffs)
    dists = np.sqrt(np.sum(sqdis, axis=1))

    unitds = diffs / dists[:, np.newaxis]
    widths = dists / partition_size

    domain = widths[:, np.newaxis] * np.arange(0, partition_size)
    domain = domain.flatten()[:, np.newaxis]

    points = np.repeat(v[:-1], partition_size, axis=0)
    unitds = np.repeat(unitds, partition_size, axis=0)

    filled = points + (unitds * domain)

    return np.vstack((filled, v[-1]))

def partition_segment(p1, p2, n_samples, endpoint=True):
    '''
    For two points in n-space, return an np.ndarray of equidistant partition
    points along the segment determined by p1 & p2.

    The total number of points returned will be n_samples. When n_samples is
    2, returns the original points.

    When endpoint is True, p2 is the last point. When false, p2 is excluded.

    Partition order is oriented from p1 to p2.

    Args:
        p1, p2:
            1 x N vectors

        partition_size:
            size of partition. should be >= 2.

    '''
    if not isinstance(n_samples, int):
        raise TypeError('partition_size should be an int.')
    elif n_samples < 2:
        raise ValueError('partition_size should be bigger than 1.')

    return (p2 - p1) * np.linspace(0, 1, num=n_samples, endpoint=endpoint)[:, np.newaxis] + p1

def partition_segment_old(p1, p2, partition_size=5):
    '''
    Deprecated. Please use partition_segment.

    For two points in n-space, return an np.ndarray of partition points at equal widths
    determined by 'partition_size' on the interior of the segment determined by p1 & p2.

    Accomplished by partitioning the segment into 'partition_size' sub-intervals.

    Partition order is oriented from p1 to p2.

    Args:
        p1, p2:
            1 x N vectors

        partition_size:
            size of partition. should be > 1.
    '''

    if not isinstance(partition_size, int):
        raise TypeError('partition_size should be an int.')
    elif partition_size < 2:
        raise ValueError('partition_size should be bigger than 1.')

    dist = np.linalg.norm(p1 - p2)

    unit_direction = (p2 - p1) / dist
    partition_width = dist / partition_size

    domain = partition_width * np.arange(1, partition_size)

    return p1 + unit_direction * domain[:, np.newaxis]

def line_intersect(p0, q0, p1, q1):
    '''
    Intersect two lines: (p0, q0) and (p1, q1). Each should be a 2D
    point.

    Adapted from http://stackoverflow.com/a/26416320/893113

    '''
    dy = q0[1] - p0[1]
    dx = q0[0] - p0[0]
    lhs0 = [-dy, dx]
    rhs0 = p0[1] * dx - dy * p0[0]

    dy = q1[1] - p1[1]
    dx = q1[0] - p1[0]
    lhs1 = [-dy, dx]
    rhs1 = p1[1] * dx - dy * p1[0]

    a = np.array([lhs0,
                  lhs1])

    b = np.array([rhs0,
                  rhs1])

    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])

def line_intersect3(p0, q0, p1, q1):
    '''
    Intersect two lines in 3d: (p0, q0) and (p1, q1). Each should be a 3D
    point.
    See this for a diagram: http://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines
    '''
    e = p0 - q0 # direction of line 0
    f = p1 - q1 # direction of line 1
    # Check for special case where we're given the intersection
    # Note that we must check for these, because if p0 == p1 then
    # g would be zero length and we can't continue
    if np.all(p0 == p1) or np.all(p0 == q1):
        return p0
    if np.all(q0 == p1) or np.all(p0 == q1):
        return q0
    g = p0 - p1 # line between to complete a triangle
    h = np.cross(f, g)
    k = np.cross(f, e)
    h_ = np.linalg.norm(h)
    k_ = np.linalg.norm(k)
    if h_ == 0 or k_ == 0:
        # there is no intesection; either parallel (k=0) or colinear (both=0) lines
        return None
    l = h_ / k_ * e
    sign = -1 if np.all(h / h_ == k / k_) else +1
    return p0 + sign * l
