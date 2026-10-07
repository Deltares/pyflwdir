# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module, specifically the wrapping of the methods which
themselves are testes elsewhere"""

import importlib
import sys

import numpy as np
import pytest
from affine import Affine

import pyflwdir
from pyflwdir import core, streams
from pyflwdir.pyflwdir import FlwdirRaster, _get_idxs_dtype

pyflwdir_module = importlib.import_module("pyflwdir.pyflwdir")
parallel_module = importlib.import_module("pyflwdir.parallel")


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
        # every cell but a pit comes after the cell it drains into
        position = np.full(flw.size, -1, dtype=np.int64)
        position[flw.idxs_seq] = np.arange(flw.ncells)
        upstream = flw.idxs_seq[flw.idxs_ds[flw.idxs_seq] != flw.idxs_seq]
        assert np.all(position[flw.idxs_ds[upstream]] < position[upstream])
        flw.order_cells(method="walk")
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
@pytest.mark.parametrize("raster", [False, True])
def test_stream_order_mask_cache(flw_real, raster):
    if raster:
        flw = FlwdirRaster(
            flw_real.idxs_ds.copy(),
            flw_real.shape,
            "d8",
            idxs_pit=flw_real.idxs_pit.copy(),
            cache=True,
        )
    else:
        flw = pyflwdir.Flwdir(
            flw_real.idxs_ds.copy(),
            idxs_pit=flw_real.idxs_pit.copy(),
            cache=True,
        )

    all_streams = flw.mask.reshape(flw.shape)
    pit_streams = np.zeros(flw.shape, dtype=bool)
    pit_streams.flat[flw.idxs_pit] = True

    all_order = flw.stream_order(mask=all_streams)
    pit_order = flw.stream_order(mask=pit_streams)
    expected = streams.strahler_order(
        flw.idxs_ds, flw.idxs_seq, mask=pit_streams.ravel()
    ).reshape(flw.shape)

    assert not np.array_equal(all_order, pit_order)
    assert np.array_equal(pit_order, expected)
    assert np.all(pit_order[~pit_streams] == 0)


@pytest.mark.unit
@pytest.mark.parametrize("raster", [False, True])
def test_add_pits_invalidates_cache(raster):
    idxs_ds = np.array([0, 0, 1, 1, core._mv], dtype=np.int32)
    idxs_pit = np.array([0], dtype=np.int32)
    if raster:
        flw = FlwdirRaster(
            idxs_ds.copy(), (1, 5), "d8", idxs_pit=idxs_pit.copy(), cache=True
        )
    else:
        flw = pyflwdir.Flwdir(idxs_ds.copy(), idxs_pit=idxs_pit.copy(), cache=True)

    old_rank = flw.rank.copy()
    old_strord = flw.stream_order().copy()
    old_idxs_us_main = flw.idxs_us_main.copy()
    if raster:
        old_distnc = flw.distnc.copy()

    flw.add_pits(idxs=np.array([2]))

    assert "rank" not in flw._cached
    assert "strord" not in flw._cached
    assert "idxs_us_main" not in flw._cached
    assert np.array_equal(
        flw.rank, core.rank(flw.idxs_ds, mv=flw._mv)[0].reshape(flw.shape)
    )
    assert not np.array_equal(flw.rank, old_rank)
    assert not np.array_equal(flw.stream_order(), old_strord)
    assert flw.idxs_us_main[1] == 3
    assert not np.array_equal(flw.idxs_us_main, old_idxs_us_main)
    if raster:
        assert "distnc" not in flw._cached
        assert not np.array_equal(flw.distnc, old_distnc)


@pytest.mark.unit
def test_repair_loops_invalidates_topology_cache():
    idxs_ds = np.array([0, 2, 1], dtype=np.int32)
    flw = pyflwdir.Flwdir(idxs_ds, idxs_pit=np.array([0], dtype=np.int32), cache=True)
    old_rank = flw.rank.copy()
    old_strord = flw.stream_order().copy()
    old_idxs_us_main = flw.idxs_us_main.copy()

    flw.repair_loops()

    assert flw.isvalid
    assert "strord" not in flw._cached
    assert "idxs_us_main" not in flw._cached
    assert np.array_equal(flw.rank, core.rank(flw.idxs_ds, mv=flw._mv)[0])
    assert not np.array_equal(flw.rank, old_rank)
    assert not np.array_equal(flw.stream_order(), old_strord)
    assert not np.array_equal(flw.idxs_us_main, old_idxs_us_main)


@pytest.mark.unit
def test_set_transform_invalidates_geometry_cache():
    idxs_ds = np.array([0, 0, 1, 1, core._mv], dtype=np.int32)
    flw = FlwdirRaster(
        idxs_ds, (1, 5), "d8", idxs_pit=np.array([0], dtype=np.int32), cache=True
    )
    old_area = flw.area.copy()
    old_distnc = flw.distnc.copy()
    old_idxs_us_main = flw.idxs_us_main.copy()

    flw.set_transform(Affine.scale(2), latlon=False)

    assert "area" not in flw._cached
    assert "distnc" not in flw._cached
    assert "idxs_us_main" not in flw._cached
    mask = flw.mask.reshape(flw.shape)
    assert np.all(flw.area[mask] == 4)
    assert not np.array_equal(flw.area, old_area)
    assert np.all(flw.distnc[mask] == 2 * old_distnc[mask])
    assert np.array_equal(flw.idxs_us_main, old_idxs_us_main)


@pytest.mark.unit
def test_main_upstream_custom_area_does_not_replace_default_cache():
    idxs_ds = np.array([0, 0, 1, 1, core._mv], dtype=np.int32)
    flw = pyflwdir.Flwdir(idxs_ds, idxs_pit=np.array([0], dtype=np.int32), cache=True)
    default_idxs_us_main = flw.idxs_us_main.copy()
    custom_uparea = np.array([1, 1, 1, 2, 0], dtype=np.float32)

    custom_idxs_us_main = flw.main_upstream(uparea=custom_uparea)

    assert custom_idxs_us_main[1] == 3
    assert default_idxs_us_main[1] == 2
    assert np.array_equal(flw.idxs_us_main, default_idxs_us_main)


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
    # a nextxy cell whose next cell is itself is a pit, and the orderings
    # start from it like from the coded pits
    nextx = np.full((3, 3), 2, dtype=np.int32)
    nexty = np.full((3, 3), 2, dtype=np.int32)
    nextx[0, 0], nexty[0, 0] = -9, -9  # a coded pit next to it
    flw = pyflwdir.from_array(np.stack([nextx, nexty]), ftype="nextxy")
    assert np.sort(flw.idxs_pit).tolist() == [0, 4]
    flw.order_cells(method="walk")
    seq_walk = flw.idxs_seq.copy()
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


@pytest.mark.integration
def test_accumulation_in_threads_matches_the_serial_one(flw_real):
    flw = pyflwdir.from_array(
        flw_real.to_array("d8"),
        ftype="d8",
        transform=flw_real.transform,
        latlon=flw_real.latlon,
        cache=False,
    )
    for method in ["asap", "cfds", "alap"]:
        layers, n_layers = flw.layer_cells(method)
        assert n_layers > 1
        assert np.count_nonzero(layers >= 0) == flw.ncells
    cfds, n_layers = flw.layer_cells("cfds")
    for layer in range(n_layers):
        members = np.flatnonzero(cfds.ravel() == layer)
        members = members[flw.idxs_ds[members] != members]
        receivers = flw.idxs_ds[members]
        assert np.unique(receivers).size == receivers.size

    data = np.random.default_rng(0).random(flw.shape)
    serial = flw.accuflux(data)
    pushed = flw.accuflux(data, parallel=True)
    assert np.allclose(pushed, serial)
    assert np.array_equal(pushed, flw.accuflux(data, parallel=True))
    import numba

    previous_threads = numba.get_num_threads()
    try:
        threaded = []
        max_threads = numba.config.NUMBA_NUM_THREADS
        thread_counts = sorted({1, min(2, max_threads), min(4, max_threads)})
        for n_threads in thread_counts:
            numba.set_num_threads(n_threads)
            threaded.append(flw.accuflux(data, parallel=True))
    finally:
        numba.set_num_threads(previous_threads)
    for result in threaded[1:]:
        assert np.array_equal(result, threaded[0])
    for layering in ["asap", "cfds", "alap"]:
        pulled = flw.accuflux(data, parallel=True, layering=layering, manner="pull")
        assert np.allclose(pulled, serial)

    assert np.array_equal(
        flw.upstream_area(unit="cell", parallel=True),
        flw.upstream_area(unit="cell"),
    )
    assert np.allclose(
        flw.upstream_area(unit="km2", parallel=True),
        flw.upstream_area(unit="km2"),
    )
    downstream = flw.accuflux(data, direction="down")
    assert downstream.shape == data.shape
    with pytest.raises(ValueError, match="only supported"):
        flw.accuflux(data, direction="down", parallel=True)
    with pytest.raises(ValueError, match="Unknown flow direction: invalid"):
        flw.accuflux(data, direction="invalid", parallel=True)
    with pytest.raises(ValueError, match="requires layering='cfds'"):
        flw.accuflux(data, parallel=True, layering="asap", manner="push")
    with pytest.raises(ValueError, match="Invalid method"):
        flw.layer_cells("invalid")


@pytest.mark.integration
def test_partition_combines_process_regions_and_thread_layers(flw_real):
    basin_parts, basin_load = flw_real.partition(n_parts=4, level="basin")
    parts, load = flw_real.partition(n_parts=4, level="subbasin")
    mask = flw_real.mask
    flat_parts = parts.ravel()
    assert np.all((flat_parts[mask] >= 0) | (flat_parts[mask] == pyflwdir.MAINSTEM))
    assert np.all(flat_parts[~mask] == -1)
    assert load.sum() + np.count_nonzero(parts == pyflwdir.MAINSTEM) == flw_real.ncells
    assert load.max() < basin_load.max()

    layers, _ = flw_real.layer_cells("cfds")
    flat_layers = layers.ravel()
    for part in range(4):
        selected = flat_parts == part
        for layer in np.unique(flat_layers[selected]):
            members = np.flatnonzero(selected & (flat_layers == layer))
            receivers = flw_real.idxs_ds[members]
            inside = selected[receivers] & (receivers != members)
            receivers = receivers[inside]
            assert np.unique(receivers).size == receivers.size

    with pytest.raises(ValueError, match="at least 1"):
        flw_real.partition(n_parts=0)
    with pytest.raises(ValueError, match="level"):
        flw_real.partition(level="invalid")


@pytest.mark.integration
def test_hybrid_process_and_thread_accumulation(flw_real):
    data = np.random.default_rng(1).random(flw_real.shape)
    serial = flw_real.accuflux(data)
    one_thread = flw_real.accuflux(
        data,
        parallel=True,
        n_processes=2,
        threads_per_process=1,
    )
    two_threads = flw_real.accuflux(
        data,
        parallel=True,
        n_processes=2,
        threads_per_process=2,
    )
    assert np.allclose(one_thread, serial)
    assert np.array_equal(two_threads, one_thread)
    basin_level = flw_real.accuflux(
        data,
        parallel=True,
        n_processes=2,
        threads_per_process=1,
        partition_level="basin",
    )
    assert np.allclose(basin_level, serial)
    assert np.array_equal(
        flw_real.upstream_area(
            unit="cell",
            parallel=True,
            n_processes=2,
            threads_per_process=1,
        ),
        flw_real.upstream_area(unit="cell"),
    )
    with pytest.raises(ValueError, match="requires parallel=True"):
        flw_real.accuflux(data, n_processes=2)
    with pytest.raises(ValueError, match="requires layering='cfds'"):
        flw_real.accuflux(
            data,
            parallel=True,
            layering="alap",
            manner="pull",
            n_processes=2,
        )


@pytest.mark.integration
def test_hybrid_spawn_reports_non_importable_main(flw_real, monkeypatch):
    monkeypatch.setattr(sys.modules["__main__"], "__file__", "<stdin>")
    with pytest.raises(RuntimeError, match="importable Python script"):
        flw_real.accuflux(
            np.ones(flw_real.shape),
            parallel=True,
            n_processes=2,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "idxs_ds",
    [
        np.array([0, 0, 1, 2, 3], dtype=np.int64),
        np.array([0, 0, 2, 2, 4, 4], dtype=np.int64),
    ],
)
def test_hybrid_accumulation_handles_a_chain_and_separate_basins(idxs_ds):
    flw = pyflwdir.Flwdir(idxs_ds)
    data = np.arange(1, idxs_ds.size + 1, dtype=np.float64)
    assert np.array_equal(
        flw.accuflux(
            data,
            parallel=True,
            n_processes=2,
            threads_per_process=1,
        ),
        flw.accuflux(data),
    )


@pytest.mark.unit
def test_partitioned_kernels_match_serial_on_random_forests():
    rng = np.random.default_rng(2)
    saw_mainstem = False
    for _ in range(20):
        size = 100
        idxs_ds = np.empty(size, dtype=np.int64)
        for idx in range(size):
            idxs_ds[idx] = rng.integers(0, idx + 1)
        flw = pyflwdir.Flwdir(idxs_ds)
        data = rng.random(size)
        serial = flw.accuflux(data)
        parts, work, merge_cells = parallel_module._plan(flw, 3, "subbasin")
        saw_mainstem |= merge_cells.size > 0
        accu = data.copy()
        for part, cells, offsets in work:
            streams.accuflux_partitioned_push(
                flw.idxs_ds,
                parts,
                part,
                cells,
                offsets,
                accu,
                -9999,
            )
        streams.accuflux_partition_mainstem(
            flw.idxs_ds,
            merge_cells,
            accu,
            -9999,
        )
        assert np.allclose(accu, serial)
    assert saw_mainstem


@pytest.mark.integration
def test_ordering_does_not_change_the_rank_ordered_methods(flw_real, flwdir_real):
    # these four build a rank-ordered sequence themselves: a tributary writes a
    # value the main stem cell beside it reads, or the elevation of one flow
    # path is read by the next, so a depth-first sequence would change what
    # they return
    out = {}
    for method in ["walk", "dfs"]:
        flw = pyflwdir.from_array(
            flw_real.to_array("d8"),
            ftype="d8",
            transform=flw_real.transform,
            latlon=flw_real.latlon,
            cache=False,
        )
        flw.order_cells(method=method)
        uparea = flw.upstream_area("km2")
        elevtn = uparea.astype(np.float32)  # any raster of floats will do here
        out[method] = (
            flw.subbasins_area(area_min=200, uparea=uparea)[0],
            flw.subbasins_pfafstetter(depth=2, uparea=uparea)[0],
            flw.dem_adjust(elevtn),
            flw.dem_dig_d4(elevtn),
        )
    for walked, dfsed in zip(out["walk"], out["dfs"]):
        assert np.array_equal(walked, dfsed)
