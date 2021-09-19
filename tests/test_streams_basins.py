# -*- coding: utf-8 -*-
"""Tests for the streams.py and basins.py submodules."""

import pytest
import numpy as np

from pyflwdir import streams, basins, core, gis_utils, regions

# import matplotlib.pyplot as plt
# parsed, flwdir = test_data[0]

# test data
from test_core import test_data


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_accuflux(parsed, flwdir):
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


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_basins(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
    n, ncol = seq.size, flwdir.shape[1]
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    # test basins
    ids = np.arange(1, idxs_pit.size + 1, dtype=int)
    bas = basins.basins(idxs_ds, idxs_pit, seq, ids)
    assert np.all(np.array([np.sum(bas == i) for i in ids]) == upa[idxs_pit])
    assert np.all(np.unique(bas[bas != 0]) == ids)  # nodata == 0
    # test region
    bas = bas.reshape(flwdir.shape)
    total_bbox = regions.region_bounds(bas)[-1]
    assert np.all(total_bbox == np.array([0, -bas.shape[0], bas.shape[1], 0]))
    areas = regions.region_area(bas)[1]
    assert areas[0] == np.sum(bas == 1)
    areas1 = regions.region_area(bas, latlon=True)[1]
    assert areas1.argmax() == areas.argmax()


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_subbasins(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
    n, ncol = seq.size, flwdir.shape[1]
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
    # # pfafstetter for largest basin
    idx0 = np.atleast_1d(idxs_pit[np.argsort(upa[idxs_pit])[-1:]])
    pfaf1 = basins.subbasins_pfafstetter(
        idx0, idxs_ds, seq, idxs_us_main, upa, mask=None, depth=1, mv=mv
    )
    assert pfaf1[idx0] == 1
    pfaf2 = basins.subbasins_pfafstetter(
        idx0, idxs_ds, seq, idxs_us_main, upa, mask=None, depth=2, mv=mv
    )
    assert pfaf2[idx0] == 11
    assert np.all(pfaf2 // 10 == pfaf1)
    pfaf_path = pfaf2[core.path(idx0, idxs_us_main, mv=mv)[0][0]]
    assert np.all(pfaf_path % 2 == 1)  # only interbasin (=odd values)
    assert np.all(np.diff(pfaf_path) >= 0)  # increasing values upstream


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_streams(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
    n, ncol = seq.size, flwdir.shape[1]
    idxs_ds[rank == -1] = mv
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    # strahler stream order
    sto = streams.strahler_order(idxs_ds, seq)
    idxs_headwater = core.headwater_indices(idxs_ds, mv=mv)
    assert np.all(sto[idxs_headwater] == 1)
    assert np.max(sto[idxs_pit]) == np.max(sto)
    assert np.all(sto[idxs_ds == mv] == 0) and np.all(sto[idxs_ds != mv] >= 1)
    # strahler stream order with mask
    sto1 = streams.strahler_order(idxs_ds, seq, mask=sto > 1)
    assert np.all(sto1[sto <= 1] == 0)
    assert np.max(sto1) == np.max(sto) - 1
    # classic stream order
    idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
    sto0 = streams.stream_order(idxs_ds, seq, idxs_us_main, mask=None, mv=mv)
    assert np.all(sto0[idxs_pit] == 1)
    assert np.max(sto0[idxs_headwater]) == np.max(sto0)
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
