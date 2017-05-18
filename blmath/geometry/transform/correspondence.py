# FIXME -- move back to core

def apply_correspondence(correspondence_src, correspondence_dst, vertices):
    """
    Apply a correspondence defined between two vertex sets to a new set.

    Identifies a correspondence between `correspondence_src` and
    `correspondence_dst` then applies that correspondence to `vertices`.
    That is, `correspondence_src` is to `correspondence_dst` as `vertices` is
    to [ return value ].

    `correspondence_src` and `vertices` must have the same topology. The return
    value will have the same topology as `correspondence_dst`. Arguments can
    be passed as `chumpy` or `numpy` arrays.

    The most common usecase here is establishing a relationship between an
    alignment and a pointcloud or set of landmarks. The pointcloud or landmarks
    can then be moved automatically as the alignment is adjusted (e.g. fit to a
    different mesh, reposed, etc).

    Args:
        correspondence_src: The source vertices for the correspondence
        correspondence_dst: The destination vertices for the correspondence
        vertices: The vertices to map using the defined correspondence

    Returns:
        the mapped version of `vertices`

    Example usage
    -------------

    >>> transformed_scan_vertices = apply_correspondence(
    ...     correspondence_src=alignment.v,
    ...     correspondence_dst=scan.v,
    ...     vertices=reposed_alignment.v
    ... )
    >>> transformed_scan = Mesh(v=transformed_scan_vertices, vc=scan.vc)

    """
    import chumpy as ch
    from bodylabs.mesh.landmarking.transformed_lm import TransformedCoeffs
    from bodylabs.mesh.landmarking.transformed_lm import TransformedLms

    ch_desired = any([
        isinstance(correspondence_src, ch.Ch),
        isinstance(correspondence_dst, ch.Ch),
        isinstance(vertices, ch.Ch),
    ])

    coeffs = TransformedCoeffs(
        src_v=correspondence_src, dst_v=correspondence_dst)

    transformed_vertices = TransformedLms(
        transformed_coeffs=coeffs, src_v=vertices)

    return transformed_vertices if ch_desired else transformed_vertices.r
