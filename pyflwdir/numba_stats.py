from numba import njit

#NOTE np.average (with) weights is not yet supoorted by numpy 
# all functions are faster than numpy.
@njit
def _average(data, weights, nodata):
    """Weighted arithmetic mean"""
    v = 0.0
    w = 0.0
    for i in range(data.size):
        v0 = data[i]
        if v0 == nodata:
            continue
        w0 = weights[i]
        v += w0 * v0
        w += w0
    if w == 0:
        return nodata
    else:
        return v / w


@njit
def _mean(data, nodata):
    """Arithmetic mean"""
    v = 0.0
    w = 0.0
    for i in range(data.size):
        v0 = data[i]
        if v0 == nodata:
            continue
        v += v0
        w += 1
    if w == 0:
        return nodata
    else:
        return v / w
