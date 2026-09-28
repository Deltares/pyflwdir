# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module, specifically the wrapping of the methods which
themselves are testes elsewhere"""

import importlib

import numpy as np
import pytest
from affine import Affine

import pyflwdir
from pyflwdir import core
from pyflwdir.pyflwdir import FlwdirRaster, _get_idxs_dtype

pyflwdir_module = importlib.import_module("pyflwdir.pyflwdir")


@pytest.mark.integration
@pytest.mark.parametrize(
    "flwdir, ftype", [("flwdir_real", "d8"), ("nextxy_real", "nextxy")]
)
def test_from_to_array(flwdir, ftype, request):
    flwdir = request.getfixturevalue(flwdir)
    mask = np.ones(flwdir.shape)
    flw = pyflwdir.from_array(flwdir, mask=mask)
    assert flw.ftype == ftype
    assert np.all(pyflwdir.from_array(flw.to_array()).idxs_ds == flw.idxs_ds)
    with pytest.raises(ValueError, match="Invalid method"):
        flw.order_cells(method="???")


@pytest.mark.unit
def test_from_array_errors(flw_real, flwdir_real):
    with pytest.raises(ValueError, match="could not be inferred."):
        pyflwdir.from_array(np.arange(20), ftype="infer")
    with pytest.raises(ValueError, match='ftype "unknown" unknown'):
        flw_real.to_array("unknown")
    with pytest.raises(ValueError, match="should be 2 dimensional"):
        pyflwdir.from_array(flwdir_real.ravel(), ftype="d8")
    with pytest.raises(ValueError, match="is invalid."):
        pyflwdir.from_array(flwdir_real, ftype="ldd", check_ftype=True)
    with pytest.raises(ValueError, match="shape does not match"):
        pyflwdir.from_array(flwdir_real, mask=np.ones((1, 1)))


@pytest.mark.unit
def test_get_idxs_dtype():
    # the smallest possible dtype is used to represent the indices
    assert _get_idxs_dtype(100) == np.int32
    assert _get_idxs_dtype(2147483647) == np.uint32
    # rasters with more than ~4.29e9 cells must use a signed dtype: a uint64
    # index dtype is promoted to float64 in numba and breaks indexing (#79)
    for n in (4294967294, 10_000_000_000):
        dtype = _get_idxs_dtype(n)
        assert np.issubdtype(dtype, np.signedinteger)
        assert np.iinfo(dtype).max >= n


@pytest.mark.unit
def test_from_array_nextxy_gets_dtype_from_cell_count(monkeypatch, nextxy_real):
    calls = []

    def get_idxs_dtype(n):
        calls.append(n)
        return np.int32

    monkeypatch.setattr(pyflwdir_module, "_get_idxs_dtype", get_idxs_dtype)

    pyflwdir.from_array(nextxy_real, ftype="nextxy")

    assert calls == [nextxy_real.shape[1] * nextxy_real.shape[2]]


@pytest.mark.unit
def test_flwdirraster_errors(flwdir_real, flwdir_real_idxs):
    idxs_ds, d8 = flwdir_real_idxs[0], flwdir_real
    with pytest.raises(ValueError, match="Unknown flow direction type"):
        pyflwdir.FlwdirRaster(idxs_ds, d8.shape, "unknown")
    with pytest.raises(ValueError, match="Invalid transform."):
        pyflwdir.FlwdirRaster(idxs_ds, d8.shape, "d8", transform=(0, 0))
    with pytest.raises(ValueError, match="Invalid FlwdirRaster: size"):
        pyflwdir.FlwdirRaster(idxs_ds[[0]], d8.shape, "d8")
    with pytest.raises(ValueError, match="Invalid FlwdirRaster: shape"):
        pyflwdir.FlwdirRaster(idxs_ds, (1, 2), "d8")
    with pytest.raises(ValueError, match="Invalid FlwdirRaster: no pits found"):
        pyflwdir.FlwdirRaster(np.array([1, 0], dtype=int), (2, 1), "d8")


def _flwdirraster_attrs_body(test_data, d8):
    idxs_ds, idxs_pit, seq, rank, mv = test_data
    for cache in [True, False]:
        flw = pyflwdir.FlwdirRaster(
            idxs_ds.copy(), d8.shape, "d8", idxs_pit=idxs_pit.copy(), cache=cache
        )
        assert flw._mv == mv
        assert flw.size == d8.size
        assert flw.shape == d8.shape
        assert isinstance(flw._dict, dict)
        assert isinstance(flw.__str__(), str)
        assert np.all(flw[flw.idxs_pit] == flw.idxs_pit)
        assert isinstance(flw.xy(flw.idxs_pit), tuple)
        assert isinstance(flw.transform, Affine)
        assert isinstance(flw.bounds, np.ndarray)
        assert np.allclose(flw.extent, flw.bounds[[0, 2, 1, 3]])
        assert isinstance(flw.latlon, bool)
        assert np.all(flw.rank.ravel() == rank)
        if cache:
            assert "rank" in flw._cached
        assert flw.ncells == seq.size
        assert np.all(np.diff(rank.flat[flw.idxs_seq]) >= 0)
        flw.repair_loops()
        assert flw.isvalid
        assert np.sum(flw.mask) == flw.ncells


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_data, flwdir",
    [("test_data_uint32", "flwdir_uint32"), ("test_data_int64", "flwdir_int64")],
)
def test_flwdirraster_attrs_unit(test_data, flwdir, request):
    _flwdirraster_attrs_body(
        request.getfixturevalue(test_data), request.getfixturevalue(flwdir)
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "test_data, flwdir",
    [("test_data_real", "flwdir_real"), ("test_data_real_int32", "flwdir_real_int32")],
)
def test_flwdirraster_attrs_integration(test_data, flwdir, request):
    _flwdirraster_attrs_body(
        request.getfixturevalue(test_data), request.getfixturevalue(flwdir)
    )


@pytest.mark.integration
def test_add_pits(flw_real, flwdir_real):
    idx0 = flw_real.idxs_pit
    x, y = flw_real.xy(flw_real.idxs_pit)
    # all cells are True -> pit at idx1
    flw_real.order_cells()  # set flw_real._seq
    flw_real.add_pits(idxs=idx0, streams=np.full(flwdir_real.shape, True, dtype=bool))
    assert np.all(flw_real.idxs_pit == idx0)
    assert flw_real._seq is None  # check if seq is deleted
    # original pit idx0
    flw_real.add_pits(xy=(x, y))
    assert np.all(flw_real.idxs_pit == idx0)
    # check some errors
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.add_pits(idxs=idx0, streams=np.ones((2, 1)))
    with pytest.raises(ValueError, match="Either idxs or xy should be provided."):
        flw_real.add_pits()
    with pytest.raises(ValueError, match="Either idxs or xy should be provided."):
        flw_real.add_pits(idxs=idx0, xy=(x, y))


# NOTE tmpdir is predefined fixture
@pytest.mark.integration
def test_save(tmpdir, flw_real):
    fn = tmpdir.join("flw_real.pkl")
    flw_real.dump(fn)
    flw1 = pyflwdir.FlwdirRaster.load(fn)
    for key in flw_real._dict:
        assert np.all(flw_real._dict[key] == flw1._dict[key])


@pytest.mark.integration
def test_path_snap(flw_real, flwdir_real_rank):
    idxs_seq = flwdir_real_rank[2]
    idx0 = idxs_seq[-1]
    # up- & downstream
    path = flw_real.path(idx0)[0]
    idx1 = flw_real.snap(idx0)[0]
    assert np.all(flw_real.path(idx1, direction="up")[0][0][::-1] == path[0])
    assert np.all(flw_real.snap(idx1, direction="up")[0] == idx0)
    assert np.all(flw_real.snap(xy=flw_real.xy(idx1), direction="up")[0] == idx0)

    # with mask
    mask = np.full(flw_real.shape, False, dtype=bool)
    path, dist = flw_real.path(idx0, mask=mask)
    idx2, _ = flw_real.snap(idx0, mask=mask)
    assert path[0].size == dist[0] + 1
    assert idx1 == idx2[0] == path[0][-1]
    # no mask
    assert np.all(path[0] == flw_real.path(idx0)[0])
    assert np.all(idx1 == flw_real.snap(idx0)[0])
    # max dist
    l = int(np.round(dist[0] / 2))
    assert l <= flw_real.path(idx0, max_length=l)[1][0] <= dist[0]
    assert l <= flw_real.snap(idx0, max_length=l)[1][0] <= dist[0]
    with pytest.raises(ValueError, match="Unknown unit"):
        flw_real.path(idx0, unit="unknown")
    with pytest.raises(ValueError, match="Unknown unit"):
        flw_real.snap(idx0, unit="unknown")
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw_real.path(idx0, direction="unknown")
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw_real.snap(idx0, direction="unknown")
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.path(idx0, mask=np.ones((2, 1)))
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.snap(idx0, mask=np.ones((2, 1)))


@pytest.mark.integration
def test_downstream(flw_real):
    idxs = np.arange(flw_real.size, dtype=int)
    assert np.all(
        flw_real.downstream(idxs).ravel()[flw_real.mask]
        == flw_real.idxs_ds[flw_real.mask]
    )
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.downstream(np.ones((2, 1)))


@pytest.mark.integration
def test_sum_upstream(flw_real):
    n_up = core.upstream_count(flw_real.idxs_ds, flw_real._mv)
    data = np.ones(flw_real.shape, dtype=np.int32)
    assert np.all(
        flw_real.upstream_sum(data).flat[flw_real.mask] == n_up[flw_real.mask]
    )
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.upstream_sum(np.ones((2, 1)))


@pytest.mark.integration
def test_moving_average(flw_real, flwdir_real_rank):
    idxs_seq = flwdir_real_rank[2]
    data = np.random.random(flw_real.shape)
    data_smooth = flw_real.moving_average(data, n=1, weights=np.ones(flw_real.shape))
    assert np.all(data_smooth == flw_real.moving_average(data, n=1))
    strord = flw_real.stream_order()
    assert np.allclose(
        flw_real.moving_average(data, n=1, restrict_strord=True),
        flw_real.moving_average(data, n=1, restrict_strord=True, strord=strord),
    )
    assert np.allclose(
        flw_real.moving_median(data, n=1, restrict_strord=True),
        flw_real.moving_median(data, n=1, restrict_strord=True, strord=strord),
    )
    idxs = flw_real.path(idxs_seq[-1], max_length=2)[0][0]
    assert np.isclose(np.mean(data.flat[idxs]), data_smooth.flat[idxs[1]])
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.moving_average(np.ones((2, 1)), n=3)
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.moving_average(data, n=5, weights=np.ones((2, 1)))


@pytest.mark.integration
def test_basins(flw_real, flwdir_real_rank):
    idxs_seq = flwdir_real_rank[2]
    # basins
    basins = flw_real.basins()
    assert basins.min() == 0
    assert basins.max() == flw_real.idxs_pit.size
    assert basins.dtype == np.uint32
    assert np.all(basins.shape == flw_real.shape)
    idx = np.arange(1, flw_real.idxs_pit.size + 1, dtype=np.int16)
    assert flw_real.basins(ids=idx).dtype == np.int16
    # subbasins
    subbasins = flw_real.basins(idxs=idxs_seq[-4:])
    assert np.any(subbasins != basins)
    # errors
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.basins(ids=np.arange(flw_real.idxs_pit.size - 1))
    with pytest.raises(ValueError, match="IDs cannot contain a value zero"):
        flw_real.basins(ids=np.zeros(flw_real.idxs_pit.size, dtype=np.int16))
    # basin bounds using IDENTITY transform
    lbs = flw_real.basin_bounds(basins)[0]
    assert np.all(lbs == np.unique(basins[basins > 0]))
    lbs, _, total_bbox = flw_real.basin_bounds(
        basins=np.ones(flw_real.shape, dtype=np.uint32)
    )
    assert np.all(np.abs(total_bbox[[1, 2]]) == flw_real.shape)
    with pytest.raises(ValueError, match="shape does not match"):
        flw_real.basin_bounds(basins=np.ones((2, 1)))
    # basin outlets
    idxs_out = flw_real.basin_outlets(basins)[1]
    assert np.all(np.sort(idxs_out) == np.sort(flw_real.idxs_pit))


@pytest.mark.integration
def test_subbasins(flw_real):
    pfaf = flw_real.subbasins_pfafstetter()[0]
    bas0 = flw_real.basins(flw_real.idxs_pit[0])
    assert np.all(pfaf[bas0 != 0] > 0)
    assert pfaf.max() <= 9
    subbas = flw_real.subbasins_streamorder()[0]
    assert np.all(subbas[bas0 != 0] > 0)
    subbas = flw_real.subbasins_area(10)[0]
    assert np.all(subbas[bas0 != 0] > 0)
    # river confluence subbasins with a 2D mask
    strord = flw_real.stream_order()
    riv_mask = strord >= (strord.max() - 2)
    subbas, idxs_out = flw_real.subbasins(riv_mask)
    assert subbas.shape == flw_real.shape
    assert subbas.dtype == np.int32
    assert idxs_out.ndim == 1 and idxs_out.size > 0
    assert np.all(
        subbas.flat[idxs_out] == np.arange(1, idxs_out.size + 1, dtype=np.int32)
    )
    assert np.all(subbas[riv_mask] > 0)


@pytest.mark.integration
def test_uparea(flw_real):
    # test with upstream grid cells
    uparea = flw_real.upstream_area()
    assert uparea.min() == -9999
    assert uparea[uparea != -9999].min() == 1
    assert uparea.dtype == np.int32
    assert np.all(uparea.shape == flw_real.shape)
    # compare with accuflux
    acc = flw_real.accuflux(np.ones(flw_real.shape))
    assert np.all(acc.flat[flw_real.mask] == uparea.flat[flw_real.mask])
    # test upstream area in km2
    uparea2 = flw_real.upstream_area(unit="km2")
    assert uparea2.dtype == np.float32
    assert uparea2.max() == uparea2.flat[flw_real.idxs_pit].max()
    with pytest.raises(ValueError, match="Unknown unit"):
        flw_real.upstream_area(unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.accuflux(np.ones((2, 1)))
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw_real.accuflux(np.ones((1, 1)), direction="???")


@pytest.mark.integration
def test_streams(flw_real, flwdir_real_rank):
    idxs_seq = flwdir_real_rank[2]
    # stream order
    strord = flw_real.stream_order()
    assert strord.flat[flw_real.mask].min() == 1
    assert strord.min() == 0
    assert strord.max() == strord.flat[flw_real.idxs_pit].max() == 5
    assert strord.dtype == np.uint8
    assert np.all(strord.shape == flw_real.shape)
    # stream segments
    feats = flw_real.streams(strord=strord)
    fstrord = np.array([f["properties"]["strord"] for f in feats])
    findex = np.array([f["properties"]["idx"] for f in feats])
    assert np.all(fstrord == strord.flat[findex])
    # check agains Flwdir
    # FIXME this fails, but only locally ??!#
    findex_ds = np.array([f["properties"]["idx_ds"] for f in feats])
    flw1 = pyflwdir.Flwdir(pyflwdir.flwdir.get_loc_idx(findex, findex_ds))
    assert np.all(fstrord == flw1.stream_order().ravel())
    # vectorize
    feats = flw_real.vectorize()
    findex = np.array([f["properties"]["idx"] for f in feats])
    assert np.all(findex == np.sort(idxs_seq))
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.geofeatures([np.array([1, 2])], xs=np.arange(3), ys=np.arange(3))
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.streams(mask=np.ones((2, 1)))
    with pytest.raises(ValueError, match="Kwargs map"):
        flw_real.geofeatures([np.array([1, 2])], uparea=np.ones((1, 1)))
    # stream distance
    data = np.zeros(flw_real.shape, dtype=np.int32)
    data[flw_real.rank > 0] = 1
    dist0 = flw_real.accuflux(data, direction="down")
    assert dist0.dtype == np.int32
    dist = flw_real.stream_distance(unit="cell")
    assert dist.max() == flw_real.rank.max()
    assert dist.dtype == np.int32
    assert np.all(dist.shape == flw_real.shape)
    assert np.all(dist0[dist != -9999] <= dist[dist != -9999])
    dist = flw_real.stream_distance(mask=np.ones(flw_real.shape, dtype=bool))
    assert np.all(dist[dist != -9999] == 0)
    dist = flw_real.stream_distance(unit="m")
    assert dist.dtype == np.float32
    with pytest.raises(ValueError, match="Unknown unit"):
        flw_real.stream_distance(unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.stream_distance(mask=np.ones((2, 1)))
    # river length
    data_smooth1 = flw_real.smooth_rivlen(data, min_rivlen=0)
    assert np.all(data_smooth1 == data)


@pytest.mark.integration
def test_upscale(flw_real, nextxy_real):
    flw1, idxs_out = flw_real.upscale(5, method="dmm")  # single method
    assert flw1.transform[0] == 5 * flw_real.transform[0]
    assert flw1.ftype == flw_real.ftype
    flwerr = flw_real.upscale_error(flw1, idxs_out)
    assert flwerr.flat[flw1.mask].min() == 0
    assert flwerr.flat[flw1.mask].max() == 1
    assert np.all(flwerr[flwerr < 0] == -1)
    with pytest.raises(ValueError, match="Unknown method"):
        flw_real.upscale(5, method="unknown")
    with pytest.raises(ValueError, match="only works for D8 or LDD"):
        pyflwdir.from_array(nextxy_real, ftype="nextxy").upscale(10)
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.upscale(5, uparea=np.ones((2, 1)))


@pytest.mark.integration
@pytest.mark.parametrize("flw", ["flw_real", "flw_real_int32"])
def test_ucat(flw, request):
    flw_real: FlwdirRaster = request.getfixturevalue(flw)
    elevtn = flw_real.rank
    hand = flw_real.hand(elevtn=elevtn, drain=elevtn == 0)
    depths = np.linspace(0.5, 1, 2)
    idxs_out = flw_real.ucat_outlets(5)
    ucat, ugrd = flw_real.ucat_area(idxs_out)
    ucat1, uvol = flw_real.ucat_volume(idxs_out, hand=hand, depths=depths)
    rivlen = flw_real.subgrid_rivlen(idxs_out)
    rivslp = flw_real.subgrid_rivslp(idxs_out, elevtn, length=1)
    rivwth = flw_real.subgrid_rivavg(idxs_out, np.ones(flw_real.shape))
    assert ugrd.shape == idxs_out.shape
    assert uvol.shape == (depths.size, *idxs_out.shape)
    assert ucat.shape == flw_real.shape
    assert np.all(ucat1 == ucat)
    assert ugrd[idxs_out != flw_real._mv].min() > 0
    assert ugrd[idxs_out != flw_real._mv].min() > 0
    assert rivlen.shape == idxs_out.shape
    assert rivlen[idxs_out != flw_real._mv].min() >= 0  # only zeros at boundary
    assert np.all(rivslp[idxs_out != flw_real._mv] > 0)
    assert np.all(rivwth[idxs_out != flw_real._mv] == 1)
    rivlen1 = flw_real.subgrid_rivlen(idxs_out=None)
    assert rivlen1.shape == flw_real.shape
    with pytest.raises(ValueError, match="Unknown method"):
        flw_real.ucat_outlets(5, method="unkown")
    with pytest.raises(ValueError, match="Unknown unit"):
        flw_real.ucat_area(idxs_out, unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.subgrid_rivslp(idxs_out, elevtn=np.ones((2, 1)))
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw_real.subgrid_rivlen(idxs_out, direction="unknown")


@pytest.mark.unit
def test_dem1():
    i = 867565
    rng = np.random.default_rng(i)
    dem = rng.random((15, 10), dtype=np.float32)
    flwdir = pyflwdir.from_dem(dem)
    dem1 = flwdir.dem_adjust(dem)
    assert np.all((dem1 - flwdir.downstream(dem1)) >= 0), i


@pytest.mark.integration
def test_dem(flw_real):
    elevtn = np.ones(flw_real.shape)
    # create values that need fix
    diff = np.logical_and(
        flw_real.rank == 2, flw_real.upstream_sum(np.ones(flw_real.shape)) >= 1
    )
    elevtn[diff] = 2.0
    elevtn_new = flw_real.dem_adjust(elevtn)
    assert np.all(elevtn_new == 1.0)
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.dem_adjust(np.ones((2, 1)))
    # hand
    rank = flw_real.rank
    drain = rank == 0
    hand = flw_real.hand(drain, elevtn_new)
    assert np.all(hand[rank > 0] == 0)
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.hand(drain, np.ones((2, 1)))
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.hand(np.ones((2, 1)), elevtn_new)
    # floodplain
    fldpln = flw_real.floodplains(elevtn_new, uparea=drain, upa_min=1, b=1)
    assert np.all(fldpln.flat[flw_real.mask] == 1)
    with pytest.raises(ValueError, match="size does not match"):
        flw_real.floodplains(np.ones((2, 1)))


@pytest.mark.unit
def test_from_array_nextxy_self_pointing_cell_is_pit():
    # a nextxy cell whose next cell is itself is a pit, and the walk ordering
    # starts from it like from the coded pits
    nextx = np.full((3, 3), 2, dtype=np.int32)
    nexty = np.full((3, 3), 2, dtype=np.int32)
    nextx[0, 0], nexty[0, 0] = -9, -9  # a coded pit next to it
    flw = pyflwdir.from_array(np.stack([nextx, nexty]), ftype="nextxy")
    assert np.sort(flw.idxs_pit).tolist() == [0, 4]
    seq_default = flw.idxs_seq.copy()  # the lazy default, i.e. the walk
    flw.order_cells(method="walk")
    seq_walk = flw.idxs_seq.copy()
    assert np.array_equal(seq_default, seq_walk)
    assert seq_walk.size == 9
    flw.order_cells(method="sort")
    assert np.array_equal(np.sort(seq_walk), np.sort(flw.idxs_seq))
    # every cell but the pits comes after its downstream cell
    position = np.full(9, -1)
    position[seq_walk] = np.arange(9)
    assert np.all(position[flw.idxs_ds[seq_walk]] <= position[seq_walk])


@pytest.mark.integration
@pytest.mark.parametrize("method", ["walk", "dfs", "topo", "sort"])
def test_order_cells_methods(flwdir_real, flwdir_real_rank, method):
    flw = pyflwdir.from_array(flwdir_real, ftype="d8")
    flw.order_cells(method=method)
    seq = flw.idxs_seq
    rank = flwdir_real_rank[0].ravel()
    # the valid cells, each after the cell it drains into
    assert np.array_equal(np.sort(seq), np.flatnonzero(rank >= 0))
    position = np.full(rank.size, -1)
    position[seq] = np.arange(seq.size)
    upstream = np.flatnonzero((rank > 0) & (flw.idxs_ds != np.arange(rank.size)))
    assert np.all(position[flw.idxs_ds[upstream]] < position[upstream])
    assert flw.ncells == seq.size
