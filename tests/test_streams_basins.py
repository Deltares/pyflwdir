# -*- coding: utf-8 -*-
"""Tests for the streams.py and basins.py submodules."""

import pytest
import numpy as np

from pyflwdir import streams, basins, core, gis_utils

_mv = core._mv

# test data
from test_core import test_data


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_uparea(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank = [p.copy() for p in parsed]
    n, ncol = seq.size, flwdir.shape[1]
    # cell count
    nodata = -9999
    material = np.full(idxs_ds.size, nodata, dtype=np.int32)
    material[seq] = 1
    acc = streams.accuflux(idxs_ds, seq, material, nodata)
    assert acc[idxs_pit].sum() == n
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    assert upa[idxs_pit].sum() == n
    assert np.all(upa == acc)
    # latlon is True
    lons, lats = gis_utils.affine_to_coords(gis_utils.IDENTITY, flwdir.shape)
    area = np.where(rank >= 0, gis_utils.reggrid_area(lats, lons).ravel(), nodata)
    acc1 = streams.accuflux(idxs_ds, seq, area, nodata)
    upa1 = streams.upstream_area(idxs_ds, seq, ncol, latlon=True)
    assert np.all(upa1 == acc1)
    # test basins
    ids = np.arange(1, idxs_pit.size + 1, dtype=np.int)
    bas = basins.basins(idxs_ds, idxs_pit, seq, ids)
    assert np.all(np.array([np.sum(bas == i) for i in ids]) == upa[idxs_pit])
    assert np.all(np.unique(bas[bas != 0]) == ids)  # nodata == 0
    # pfafstetter
    idx0 = idxs_pit[np.argsort(upa[idxs_pit])[-1]]
    pfaf = basins.pfafstetter(idx0, idxs_ds, seq, upa, upa_min=0, depth=2)
    if pfaf.max() > 10:
        pfaf = pfaf // 10
    pfafmax = pfaf.max()
    assert pfaf[idx0] == 1
    upa_out = np.array([np.max(upa[pfaf == i]) for i in range(1, pfafmax + 1)])
    # inter basins ordered from down- to upstream
    assert np.all(np.diff(upa_out[np.arange(0, pfafmax, 2)]) <= 0)
    # subbasins outlet smaller than next interbasin outlet
    subbas_upa_out = upa_out[np.arange(1, pfafmax, 2)]
    intbas_upa_out = upa_out[np.arange(2, pfafmax, 2)]
    assert np.all((subbas_upa_out - intbas_upa_out) <= 0)


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_stream(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank = [p.copy() for p in parsed]
    n, ncol = seq.size, flwdir.shape[1]
    idxs_ds[rank == -1] = _mv
    # stream order
    sto = streams.stream_order(idxs_ds, seq)
    idxs_headwater = core.headwater_indices(idxs_ds)
    assert np.all(sto[idxs_headwater] == 1)
    assert np.max(sto[idxs_pit]) == np.max(sto)
    assert np.all(sto[idxs_ds == _mv] == -1) and np.all(sto[idxs_ds != _mv] >= 1)
    # stream distance
    strlen = streams.stream_distance(idxs_ds, seq, ncol)
    assert np.all(strlen[idxs_pit] == 0)
    assert np.max(strlen[idxs_headwater]) == np.max(strlen)
    ranks1 = streams.stream_distance(idxs_ds, seq, ncol, real_length=False)
    assert np.all(ranks1[rank >= 0] == rank[rank >= 0])
