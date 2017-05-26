import numpy as np
from blmath.util.decorators import setter_property

class Polyline(object):
    '''
    Represent the geometry of a polygonal chain in 3-space. The
    chain may be open or closed, and there are no constraints on the
    geometry. For example, the chain may be simple or
    self-intersecting, and the points need not be unique.

    Mutable by setting polyline.v or polyline.closed or calling
    a method like polyline.partition_by_length().

    This replaces the functions in blmath.geometry.segment.

    Note this class is distinct from lace.lines.Lines, which
    allows arbitrary edges and enables visualization. To convert to
    a Lines object, use the as_lines() method.

    '''
    def __init__(self, v, closed=False):
        '''
        v: An array-like thing containing points in 3-space.
        closed: True indicates a closed chain, which has an extra
          segment connecting the last point back to the first
          point.

        '''
        # Avoid invoking _update_edges before setting closed and v.
        self.__dict__['closed'] = closed
        self.v = v

    def __repr__(self):
        if self.v is not None and len(self.v) != 0:
            if self.closed:
                return "<closed Polyline with {} verts>".format(len(self))
            else:
                return "<open Polyline with {} verts>".format(len(self))
        else:
            return "<Polyline with no verts>"

    def __len__(self):
        return len(self.v)

    def copy(self):
        '''
        Return a copy of this polyline.

        '''
        v = None if self.v is None else np.copy(self.v)
        return self.__class__(v, closed=self.closed)

    def as_lines(self, vc=None):
        '''
        Return a Lines instance with our vertices and edges.

        '''
        from lace.lines import Lines
        return Lines(v=self.v, e=self.e, vc=vc)

    def to_dict(self, decimals=3):
        return {
            'vertices': [np.around(v, decimals=decimals).tolist() for v in self.v],
            'edges': self.e,
        }

    def _update_edges(self):
        if self.v is None:
            self.__dict__['e'] = None
            return

        num_vertices = self.v.shape[0]
        num_edges = num_vertices if self.closed else num_vertices - 1

        edges = np.vstack([np.arange(num_edges), np.arange(num_edges) + 1]).T

        if self.closed:
            edges[-1][1] = 0

        edges.flags.writeable = False

        self.__dict__['e'] = edges

    @setter_property
    def v(self, val):  # setter_property incorrectly triggers method-hidden. pylint: disable=method-hidden
        '''
        Update the vertices to a new array-like thing containing points
        in 3D space. Set to None for an empty polyline.

        '''
        from blmath.numerics import as_numeric_array
        self.__dict__['v'] = as_numeric_array(val, dtype=np.float64, shape=(-1, 3), allow_none=True)
        self._update_edges()

    @setter_property
    def closed(self, val):
        '''
        Update whether the polyline is closed or open.

        '''
        self.__dict__['closed'] = val
        self._update_edges()

    @property
    def e(self):
        '''
        Return a np.array of edges. Derived automatically from self.v
        and self.closed whenever those values are set.

        '''
        return self.__dict__['e']

    @property
    def segment_lengths(self):
        '''
        The length of each of the segments.

        '''
        if self.e is None:
            return np.empty((0,))

        return ((self.v[self.e[:, 1]] - self.v[self.e[:, 0]]) ** 2.0).sum(axis=1) ** 0.5

    @property
    def total_length(self):
        '''
        The total length of all the segments.

        '''
        return np.sum(self.segment_lengths)

    def partition_by_length(self, max_length, ret_indices=False):
        '''
        Subdivide each line segment longer than max_length with
        equal-length segments, such that none of the new segments
        are longer than max_length.

        ret_indices: If True, return the indices of the original vertices.
          Otherwise return self for chaining.

        '''
        from blmath.geometry.segment import partition_segment

        lengths = self.segment_lengths
        num_segments_needed = np.ceil(lengths / max_length)

        indices_of_orig_vertices = []
        new_v = np.empty((0, 3))

        for i, num_segments in enumerate(num_segments_needed):
            start_point, end_point = self.v[self.e[i]]

            indices_of_orig_vertices.append(len(new_v))

            # In the simple case, one segment, or degenerate case, with
            # a repeated vertex, we do not need to subdivide.
            if num_segments <= 1:
                new_v = np.vstack((new_v, start_point.reshape(-1, 3)))
            else:
                new_v = np.vstack((new_v, partition_segment(start_point, end_point, np.int(num_segments), endpoint=False)))

        if not self.closed:
            indices_of_orig_vertices.append(len(new_v))
            new_v = np.vstack((new_v, self.v[-1].reshape(-1, 3)))

        self.v = new_v

        return np.array(indices_of_orig_vertices) if ret_indices else self

    def apex(self, axis):
        '''
        Find the most extreme point in the direction of the axis provided.

        axis: A vector, which is an 3x1 np.array.

        '''
        from blmath.geometry.apex import apex
        return apex(self.v, axis)
