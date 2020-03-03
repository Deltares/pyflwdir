from numba import njit

# NOTE np.average (with) weights is not yet supoorted by numpy
# all functions are faster than numpy.
@njit
def _average(data, weights, nodata):
    """Weighted arithmetic mean
    NOTE: does not work with nodata=np.nan!
    """
    v = 0.0
    w = 0.0
    for i in range(data.size):
        v0 = data[i]
        if v0 == nodata:
            continue
        w0 = weights[i]
        v += w0 * v0
        w += w0
    return v / w if w != 0 else nodata


@njit
def _mean(data, nodata):
    """Arithmetic mean
    NOTE: does not work with nodata=np.nan!
    """
    v = 0.0
    w = 0.0
    for v0 in data:
        if v0 == nodata:
            continue
        v += v0
        w += 1.0
    return v / w if w != 0 else nodata
