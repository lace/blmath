def scale_to_range(a, r_min, r_max):
    import numpy as np
    a = a - np.min(a)
    return (r_max - r_min)*a/np.max(a) + r_min

def zero_safe_divide(a, b, default_error_value=0.):
    """Element-wise division that accounts for floating point errors.

    Both invalid floating-point (e.g. 0. / 0.) and divide be zero errors are
    suppressed. Resulting values (NaN and Inf respectively) are replaced with
    `default_error_value`.

    """
    import numpy as np

    with np.errstate(invalid='ignore', divide='ignore'):
        quotient = np.true_divide(a, b)
        bad_value_indices = np.logical_or(
            np.isnan(quotient), np.isinf(quotient))
        quotient[bad_value_indices] = default_error_value

    return quotient
