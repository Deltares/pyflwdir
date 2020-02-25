from numba import njit
from numba.typed import List
import numpy as np
import pandas as pd
from scipy import ndimage

from pyflwdir import core, gis_utils

__all__ = ["mean_drainage_path_stats"]


@njit
def _drainage_paths(tree, idxs_us, rivlen):
    """returns indices and length of drainage paths 
    (from outlet to most upstream)."""
    # intialize map
    basins = np.zeros(rivlen.size, dtype=np.uint32)
    updist = np.zeros(rivlen.size, dtype=rivlen.dtype)
    idxs_ds0 = tree[0]
    basins[idxs_ds0] = np.array([i for i in range(idxs_ds0.size)], dtype=np.uint32) + 1
    updist[idxs_ds0] = rivlen[idxs_ds0]
    idx_lst = []
    upd_lst = []
    bas_lst = []
    # loop through flwdir map
    for i in range(len(tree)):
        idxs_ds0 = tree[i]  # from down- to upstream
        for idx_ds in idxs_ds0:
            idxs_us0 = idxs_us[idx_ds, :]  # NOTE: contains _mv values
            d = updist[idx_ds]
            b = basins[idx_ds]
            for ii in range(idxs_us0.size):
                idx_us = idxs_us0[ii]
                if idx_us == core._mv:
                    break
                updist[idx_us] = d + rivlen[idx_us]
                basins[idx_us] = b
            if ii == 0:  # most upstream
                idx_lst.append(idx_ds)
                upd_lst.append(d)
                bas_lst.append(b)
    return np.array(idx_lst), np.array(upd_lst), np.array(bas_lst)


def mean_drainage_path_stats(tree, idxs_us, rivlen, elevtn):
    idxs_ds0 = tree[0]
    index = np.arange(idxs_ds0.size, dtype=np.uint32) + 1
    idxs_us0, drain_len, basins = _drainage_paths(tree, idxs_us, rivlen)
    drain_slp = (elevtn[idxs_us0] - elevtn[idxs_ds0[basins - 1]]) / drain_len
    assert np.all(np.unique(basins) == index)
    df_out = pd.DataFrame(index=index)
    df_out["mean_drain_length"] = ndimage.mean(drain_len, basins, index=index)
    df_out["mean_drain_slope"] = ndimage.mean(drain_slp, basins, index=index)
    return df_out
