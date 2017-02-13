def np_reshape_safe(A, shape, just_warn=False):
    '''
    ndarray.reshape has the potential to create a copy. This will will raise an exception or issue
    a warning if a copy would be made.
    '''
    import warnings
    res = A.view()
    if just_warn:
        try:
            res.shape = shape
        except AttributeError:
            warnings.warn("Reshape made a copy rather than just returning a view", stacklevel=2)
            res = A.reshape(shape)
    else:
        res.shape = shape
    return res

def np_make_readonly(arr):
    import numpy as np

    for a in arr:
        if isinstance(a, np.ndarray):
            a.flags.writeable = False
