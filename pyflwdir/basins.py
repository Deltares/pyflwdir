# -*- coding: utf-8 -*-
"""Methods to delineate (sub)basins."""
from numba import njit
import numpy as np

from pyflwdir import core

_mv = core._mv
all = []


@njit
def fillnodata_upstream(idxs_ds, seq, data, nodata):
    """Retuns a a copy of <data> where upstream cell with <nodata> values are filled 
    based on the first downstream valid cell value. 
    
    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    data : 1D array
        original data with missing values
    nodata : float, integer
        nodata value
    
    Returns
    -------
    1D array of data.dtype
        infilled data
    """
    res = data.copy()
    for idx0 in seq:  # down- to upstream
        if res[idx0] == nodata:
            res[idx0] = res[idxs_ds[idx0]]
    return res


def basins(idxs_ds, idxs_pit, seq, ids=None):
    """return basin map"""
    if ids is None:
        ids = np.arange(1, idxs_pit.size + 1, dtype=np.uint32)
    basins = np.zeros(idxs_ds.size, dtype=ids.dtype)
    basins[idxs_pit] = ids
    return fillnodata_upstream(idxs_ds, seq, basins, 0)


# TODO check
# @njit # NOTE does not work atm with dicts (numba 0.48)
def pfafstetter(idx0, idxs_ds, seq, uparea, upa_min=0, depth=1, mv=_mv):
    """pfafstetter coding for single basin
    
    Verdin K . and Verdin J . 1999 A topological system for delineation 
    and codification of the Earth’s river basins J. Hydrol. 218 1–12 
    Online: https://linkinghub.elsevier.com/retrieve/pii/S0022169499000116
    """
    #
    idxs_us_main = core.main_upstream(idxs_ds, uparea, upa_min, mv=mv)
    idxs_us_trib = core.main_tributary(idxs_ds, idxs_us_main, uparea, upa_min, mv=mv)
    #
    upa_min = np.atleast_1d(upa_min)
    pfaf = np.zeros(uparea.size, dtype=np.uint32)
    # initialize
    pfafs = np.array([0], dtype=pfaf.dtype)
    pfaf_dict = dict()
    idx1 = mv
    for d in range(depth):
        pfaf_lst_next = []
        min_upa0 = upa_min[min(upa_min.size - 1, d)]
        for base in pfafs:
            if d > 0:
                i = base % 10 - 1
                pfafid = base * 10
                idxs0 = pfaf_dict[base // 10]
                idx0 = idxs0[i]
                idx1 = idxs0[i + 2] if i % 2 == 0 and i < idxs0.size - 1 else mv
            else:
                pfafid = 0
            idxs_sub = core._tributaries(
                idx0, idxs_us_main, idxs_us_trib, uparea, idx1, min_upa0, 4, mv=mv
            )
            idxs_sub = idxs_sub[idxs_sub != mv]
            if idxs_sub.size > 0:
                idxs_inter = idxs_us_main[idxs_ds[idxs_sub]]
                idxs1 = [idx0]
                for i in range(idxs_sub.size):
                    idxs1.append(idxs_sub[i])
                    idxs1.append(idxs_inter[i])
                idxs1 = np.array(idxs1)
                pfaf_lst = [pfafid + k for k in range(1, idxs1.size + 1)]
                pfaf_lst_next.extend(pfaf_lst)
                pfaf[idxs1] = np.array(pfaf_lst, dtype=pfaf.dtype)
                pfaf_dict[base] = idxs1
            elif d > 0:
                pfaf[idx0] = pfafid + 1
        pfafs = np.array(pfaf_lst_next, dtype=pfaf.dtype)
    return fillnodata_upstream(idxs_ds, seq, pfaf, 0)
