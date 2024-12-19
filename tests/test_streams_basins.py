# -*- coding: utf-8 -*-
"""Tests for the streams.py and basins.py submodules."""

import pytest
import numpy as np

from pyflwdir import streams, basins, core, gis_utils, regions


@pytest.mark.parametrize(
    "test_data, flwdir",
    [("test_data0", "flwdir0"), ("test_data1", "flwdir1"), ("test_data2", "flwdir2")],
)
def test_accuflux(test_data, flwdir, request):
    flwdir = request.getfixturevalue(flwdir)
    test_data = request.getfixturevalue(test_data)
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in test_data]
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


@pytest.mark.parametrize(
    "test_data, flwdir",
    [("test_data0", "flwdir0"), ("test_data1", "flwdir1"), ("test_data2", "flwdir2")],
)
def test_basins(test_data, flwdir, request):
    flwdir = request.getfixturevalue(flwdir)
    test_data = request.getfixturevalue(test_data)
    idxs_ds, idxs_pit, seq, _, _ = [p.copy() for p in test_data]
    _, ncol = seq.size, flwdir.shape[1]
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
    lbs, areas = regions.region_area(bas)
    assert areas[0] == np.sum(bas == 1)
    areas1 = regions.region_area(bas, latlon=True)[1]
    assert areas1.argmax() == areas.argmax()
    # test dissolve with labels
    lbs0 = lbs[np.argmin(areas)]
    bas1 = regions.region_dissolve(bas, labels=lbs0)
    assert np.all(~np.isin(bas1, lbs0))
    # test dissovle with linear index
    idxs = idxs_pit[np.argsort(upa[idxs_pit])][:2]
    lbs0 = bas.flat[idxs]
    bas1 = regions.region_dissolve(bas, idxs=idxs)
    assert np.all(~np.isin(bas1, lbs0))
    # dissolve errors
    with pytest.raises(ValueError, match='Either "labels" or "idxs" must be provided'):
        regions.region_dissolve(bas)
    with pytest.raises(ValueError, match="Found non-unique or zero-value labels"):
        regions.region_dissolve(bas, labels=0)


@pytest.mark.parametrize(
    "test_data, flwdir",
    [("test_data0", "flwdir0"), ("test_data1", "flwdir1"), ("test_data2", "flwdir2")],
)
def test_subbasins(test_data, flwdir, request):
    flwdir = request.getfixturevalue(flwdir)
    test_data = request.getfixturevalue(test_data)
    idxs_ds, idxs_pit, seq, _, mv = [p.copy() for p in test_data]
    _, ncol = seq.size, flwdir.shape[1]
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
    ## pfafstetter for largest basin
    idx0 = np.atleast_1d(idxs_pit[np.argsort(upa[idxs_pit])[-1:]])
    pfaf1, idxs_out1 = basins.subbasins_pfafstetter(
        idx0, idxs_ds, seq, idxs_us_main, upa, mask=None, depth=1, mv=mv
    )
    assert pfaf1[idx0] == 1
    lbs, idxs_out = regions.region_outlets(pfaf1.reshape(flwdir.shape), idxs_ds, seq)
    idxs_out1 = idxs_out1[np.argsort(pfaf1[idxs_out1])]
    assert np.all(idxs_out1 == idxs_out)
    assert np.all(lbs == pfaf1[idxs_out1])
    pfaf2, _ = basins.subbasins_pfafstetter(
        idx0, idxs_ds, seq, idxs_us_main, upa, mask=None, depth=2, mv=mv
    )
    assert pfaf2[idx0] == 11
    assert np.all(pfaf2 // 10 == pfaf1)
    pfaf_path = pfaf2[core.path(idx0, idxs_us_main, mv=mv)[0][0]]
    assert np.all(pfaf_path % 2 == 1)  # only interbasin (=odd values)
    assert np.all(np.diff(pfaf_path) >= 0)  # increasing values upstream
    ## area subbasins
    subbas, idxs_out1 = basins.subbasins_area(
        idxs_ds, seq, idxs_us_main, upa, area_min=5
    )
    assert np.all(upa[subbas == 0] == -9999)
    pits = idxs_ds[idxs_out1] == idxs_out1
    assert np.all(subbas[idxs_out1][~pits] != subbas[idxs_ds[idxs_out1]][~pits])
    lbs0 = subbas[idxs_out1][~pits]
    lbs, areas = regions.region_area(subbas.reshape(flwdir.shape))
    # all nonpits must have area_min size
    assert np.all(areas[np.isin(lbs, lbs0)] > 5)


@pytest.mark.parametrize("test_data", ["test_data0", "test_data1"])
def test_subbasins_strord(test_data, request):
    test_data = request.getfixturevalue(test_data)
    idxs_ds, _, seq, _, _ = [p.copy() for p in test_data]
    ## streamorder basins
    strord = streams.strahler_order(idxs_ds, seq)
    maxsto = strord.max()
    subbas, idxs_out1 = basins.subbasins_streamorder(idxs_ds, seq, strord, min_sto=-2)
    sto_out = strord[idxs_out1]
    assert np.all(sto_out >= maxsto - 2)
    assert np.all(strord[subbas == 0] < maxsto - 2)
    pits = idxs_ds[idxs_out1] == idxs_out1
    sto_out1 = strord[idxs_ds[idxs_out1]]
    assert np.all(sto_out1[~pits] > sto_out[~pits])
    assert np.all(subbas[idxs_out1][~pits] != subbas[idxs_ds[idxs_out1]][~pits])


@pytest.mark.parametrize(
    "test_data, flwdir",
    [("test_data0", "flwdir0"), ("test_data1", "flwdir1"), ("test_data2", "flwdir2")],
)
def test_streams(test_data, flwdir, request):
    flwdir = request.getfixturevalue(flwdir)
    test_data = request.getfixturevalue(test_data)
    idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in test_data]
    _, ncol = seq.size, flwdir.shape[1]
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


def test_smooth_rivlen(test_data0, flwdir0):
    idxs_ds, _, seq, _, mv = test_data0
    ncol = flwdir0.shape[1]
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
    rivlen = np.random.rand(idxs_ds.size)
    rivlen[upa <= 3] = -9999.0  # river cells with at least 3 upstream cells
    min_rivlen = 0.2
    rivlen_out = streams.smooth_rivlen(
        idxs_ds,
        idxs_us_main,
        rivlen,
        min_rivlen=min_rivlen,
        max_window=10,
        nodata=-9999.0,
        mv=mv,
    )
    # NOTE: there could still be cells with rivlen < min_rivlen
    assert rivlen_out[rivlen_out < min_rivlen].size < rivlen[rivlen < min_rivlen].size
    assert np.isclose(np.sum(rivlen_out[rivlen_out > 0]), np.sum(rivlen[rivlen > 0]))
