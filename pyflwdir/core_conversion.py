# -*- coding: utf-8 -*-
"""Methods to convert between different flwdir types"""

from numba import njit
import numpy as np
from pyflwdir import core_d8, core_ldd

__all__ = ["d8_to_ldd", "ldd_to_d8"]


def d8_to_ldd(flwdir):
    """Return ldd based on d8 array."""
    # create conversion dict
    remap = {k: v for (k, v) in zip(core_d8._ds.flatten(), core_ldd._ds.flatten())}
    # add addional land pit code to pcr pit
    remap.update({core_d8._pv[1]: core_ldd._pv, core_d8._mv: core_ldd._mv})
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_ldd._mv))(flwdir)


def ldd_to_d8(flwdir):
    """Return d8 based on ldd array."""
    # create conversion dict
    remap = {k: v for (k, v) in zip(core_ldd._ds.flatten(), core_d8._ds.flatten())}
    # add addional land pit code to pcr pit
    remap.update({core_ldd._pv: core_d8._pv[0], core_ldd._mv: core_d8._mv})
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_d8._mv))(flwdir)


@njit
def _local_d4(idx0, idx_ds, ncol):
    """Return indices of d4 neighbors, e.g.: indices of N, W neigbors if flowdir is NW."""
    idxs_d4 = [
        idx0 - ncol,
        idx0 - 1,
        idx0 + ncol,
        idx0 + 1,
        idx0 - ncol,
    ]  # n, w, s, e, n
    idxs_diag = [
        idx0 - ncol - 1,
        idx0 + ncol - 1,
        idx0 + ncol + 1,
        idx0 - ncol + 1,
    ]  # nw, sw, se, ne
    di = idxs_diag.index(idx_ds)
    return np.asarray(idxs_d4[di : di + 2])


@njit
def to_d4(idxs_ds, seq, elv_flat, shape):
    """Return indices for d4 flow direction based on initial d8 flow directions
    where diagonal flow directions are modified d4 based on minimal difference in elevation.
    """
    idxs_ds_d4 = idxs_ds.copy()
    _, ncol = shape
    msk_up = np.ones(idxs_ds.size, dtype=np.bool)
    msk_up[seq] = False
    for idx0 in seq[::-1]:  # up- to downstream
        if msk_up[idx0]:
            continue
        idx_ds = idxs_ds[idx0]
        dd = abs(idx0 - idx_ds)
        msk_up[idx0] = True
        if dd <= 1 or dd == ncol:  # D4
            continue
        d4 = _local_d4(idx0, idx_ds, ncol, msk_up)
        d4 = d4[msk_up[d4]]
        if len(d4) > 0:
            idx_ds1 = idxs_ds[idx_ds]
            dd1 = abs(idx_ds - idx_ds1)
            if not (dd1 <= 1 or dd1 == ncol):  # next cell also diag
                d4_1 = _local_d4(idx_ds, idx_ds1, ncol)
                d4_1 = d4_1[msk_up[d4_1]]
                d4_s = d4[[i in d4_1 for i in d4]]
                if d4_s:
                    dz_s = abs(elv_flat[d4_s] - elv_flat[idx0])
                    dz_1 = min(abs(elv_flat[d4] - elv_flat[idx0]))
                    dz_2 = min(abs(elv_flat[d4_1] - elv_flat[idx_ds]))
                    if dz_s < (dz_1 + dz_2):
                        idxs_ds_d4[idx0] = d4_s
                        idxs_ds_d4[d4_s] = idx_ds1
                        idxs_ds_d4[idx_ds] = -1
                        msk_up[idx_ds] = True
                        continue
            d4_min = d4[abs(np.argmin(elv_flat[d4] - elv_flat[idx0]))]
            idxs_ds_d4[idx0] = d4_min
            idxs_ds_d4[d4_min] = idx_ds
        else:
            print("error")
            break
    return idxs_ds_d4
