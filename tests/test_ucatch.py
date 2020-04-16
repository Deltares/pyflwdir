# -*- coding: utf-8 -*-
"""Tests for the unitcatchments.py submodule"""

import pytest
import numpy as np

from pyflwdir import unitcatchments as ucat
from pyflwdir import core, streams

# test data
from test_core import test_data

_mv = core._mv
parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, ranks = parsed
ncol, shape = flwdir.shape[1], flwdir.shape
upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
elv = ranks

test = [("eam", 5), ("none", 1), ("dmm", 4)]


@pytest.mark.parametrize("method, cellsize", test)
def test_ucatch(method, cellsize):
    if cellsize == 1:
        idxs_out = np.arange(idxs_ds.size)
        idxs_out[idxs_ds == _mv] = _mv
        shape_out = shape
    else:
        idxs_out, shape_out = ucat.outlets(idxs_ds, upa, cellsize, shape, method=method)
    umap, uare = ucat.unit_catchments(idxs_out, idxs_ds, seq, ncol, dtype=np.int32)
    rivlen, rivslp = ucat.channel_length_slope(
        idxs_out, idxs_ds, upa, elv, ncol, latlon=True
    )
    if cellsize == 1:
        assert np.all(uare[umap != -1] == cellsize)
        assert np.all(rivlen[upa == 1] == 0)  # headwater cells
        assert np.all(rivlen[upa > 1] >= 1)  # downstream cells
        # dz == 1 (elv == ranks)
        assert np.all(rivslp[rivlen > 0] == 1 / rivlen[rivlen > 0])
    assert np.all(rivslp[idxs_out != -1] >= 0)
    assert umap.max() == np.where(idxs_out != _mv)[0][-1]
    assert np.all(uare[idxs_out != -1] >= 1)
