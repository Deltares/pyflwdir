# -*- coding: utf-8 -*-
"""Tests for the streams.py and basins.py submodules."""

import pytest
import numpy as np

from pyflwdir import streams, basins, core, gis_utils, regions

# test data
from test_core import test_data


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_streams_basins_up(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
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
    # test region
    bas = bas.reshape(flwdir.shape)
    total_bbox = regions.region_bounds(bas)[-1]
    assert np.all(total_bbox == np.array([0, 0, bas.shape[1], bas.shape[0]]))
    areas = regions.region_area(bas)[1]
    assert areas[0] == np.sum(bas == 1)
    areas1 = regions.region_area(bas, latlon=True)[1]
    assert areas1.argmax() == areas.argmax()
    # pfafstetter
    idx0 = idxs_pit[np.argsort(upa[idxs_pit])[-1:]]
    pfaf = basins.pfafstetter(idx0, idxs_ds, seq, upa, upa_min=0, depth=2, mv=mv)
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
def test_streams_basins_ds(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
    n, ncol = seq.size, flwdir.shape[1]
    idxs_ds[rank == -1] = mv
    # stream order
    sto = streams.stream_order(idxs_ds, seq)
    idxs_headwater = core.headwater_indices(idxs_ds, mv=mv)
    assert np.all(sto[idxs_headwater] == 1)
    assert np.max(sto[idxs_pit]) == np.max(sto)
    assert np.all(sto[idxs_ds == mv] == -1) and np.all(sto[idxs_ds != mv] >= 1)
    # stream distance
    data = np.zeros(idxs_ds.size, dtype=np.int32)
    data[rank > 0] = 1
    strlen0 = streams.accuflux_ds(idxs_ds, seq, data, -1)
    assert np.all(rank[rank >= 0] == strlen0[rank >= 0])
    strlen = streams.stream_distance(idxs_ds, seq, ncol)
    assert np.all(strlen[idxs_pit] == 0)
    assert np.max(strlen[idxs_headwater]) == np.max(strlen)
    ranks1 = streams.stream_distance(idxs_ds, seq, ncol, real_length=False)
    assert np.all(ranks1[rank >= 0] == rank[rank >= 0])
