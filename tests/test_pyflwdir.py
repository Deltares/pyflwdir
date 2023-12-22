# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module, specifically the wrapping of the methods which
themselves are testes elsewhere"""

import numpy as np
import pytest
from affine import Affine

import pyflwdir
from pyflwdir import core


@pytest.mark.parametrize("flwdir, ftype", [("flwdir0", "d8"), ("nextxy0", "nextxy")])
def test_from_to_array(flwdir, ftype, request):
    flwdir = request.getfixturevalue(flwdir)
    mask = np.ones(flwdir.shape)
    flw = pyflwdir.from_array(flwdir, mask=mask)
    assert flw.ftype == ftype
    assert np.all(pyflwdir.from_array(flw.to_array()).idxs_ds == flw.idxs_ds)
    with pytest.raises(ValueError, match="Invalid method"):
        flw.order_cells(method="???")


def test_from_array_errors(flw0, flwdir0):
    with pytest.raises(ValueError, match="could not be inferred."):
        pyflwdir.from_array(np.arange(20), ftype="infer")
    with pytest.raises(ValueError, match='ftype "unknown" unknown'):
        flw0.to_array("unknown")
    with pytest.raises(ValueError, match="should be 2 dimensional"):
        pyflwdir.from_array(flwdir0.ravel(), ftype="d8")
    with pytest.raises(ValueError, match="is invalid."):
        pyflwdir.from_array(flwdir0, ftype="ldd", check_ftype=True)
    with pytest.raises(ValueError, match="shape does not match"):
        pyflwdir.from_array(flwdir0, mask=np.ones((1, 1)))


def test_flwdirraster_errors(flwdir0, flwdir0_idxs):
    idxs_ds, d8 = flwdir0_idxs[0], flwdir0
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


@pytest.mark.parametrize(
    "test_data, flwdir", [("test_data0", "flwdir0"), ("test_data1", "flwdir1")]
)
def test_flwdirraster_attrs(test_data, flwdir, request):
    d8 = request.getfixturevalue(flwdir)
    test_data = request.getfixturevalue(test_data)
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


def test_add_pits(flw0, flwdir0):
    idx0 = flw0.idxs_pit
    x, y = flw0.xy(flw0.idxs_pit)
    # all cells are True -> pit at idx1
    flw0.order_cells()  # set flw0._seq
    flw0.add_pits(idxs=idx0, streams=np.full(flwdir0.shape, True, dtype=bool))
    assert np.all(flw0.idxs_pit == idx0)
    assert flw0._seq is None  # check if seq is deleted
    # original pit idx0
    flw0.add_pits(xy=(x, y))
    assert np.all(flw0.idxs_pit == idx0)
    # check some errors
    with pytest.raises(ValueError, match="size does not match"):
        flw0.add_pits(idxs=idx0, streams=np.ones((2, 1)))
    with pytest.raises(ValueError, match="Either idxs or xy should be provided."):
        flw0.add_pits()
    with pytest.raises(ValueError, match="Either idxs or xy should be provided."):
        flw0.add_pits(idxs=idx0, xy=(x, y))


# NOTE tmpdir is predefined fixture
def test_save(tmpdir, flw0):
    fn = tmpdir.join("flw0.pkl")
    flw0.dump(fn)
    flw1 = pyflwdir.FlwdirRaster.load(fn)
    for key in flw0._dict:
        assert np.all(flw0._dict[key] == flw1._dict[key])


def test_path_snap(flw0, flwdir0_rank):
    idxs_seq = flwdir0_rank[2]
    idx0 = idxs_seq[-1]
    # up- & downstream
    path = flw0.path(idx0)[0]
    idx1 = flw0.snap(idx0)[0]
    assert np.all(flw0.path(idx1, direction="up")[0][0][::-1] == path[0])
    assert np.all(flw0.snap(idx1, direction="up")[0] == idx0)
    assert np.all(flw0.snap(xy=flw0.xy(idx1), direction="up")[0] == idx0)

    # with mask
    mask = np.full(flw0.shape, False, dtype=bool)
    path, dist = flw0.path(idx0, mask=mask)
    idx2, _ = flw0.snap(idx0, mask=mask)
    assert path[0].size == dist[0] + 1
    assert idx1 == idx2[0] == path[0][-1]
    # no mask
    assert np.all(path[0] == flw0.path(idx0)[0])
    assert np.all(idx1 == flw0.snap(idx0)[0])
    # max dist
    l = int(np.round(dist[0] / 2))
    assert l <= flw0.path(idx0, max_length=l)[1][0] <= dist[0]
    assert l <= flw0.snap(idx0, max_length=l)[1][0] <= dist[0]
    with pytest.raises(ValueError, match="Unknown unit"):
        flw0.path(idx0, unit="unknown")
    with pytest.raises(ValueError, match="Unknown unit"):
        flw0.snap(idx0, unit="unknown")
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw0.path(idx0, direction="unknown")
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw0.snap(idx0, direction="unknown")
    with pytest.raises(ValueError, match="size does not match"):
        flw0.path(idx0, mask=np.ones((2, 1)))
    with pytest.raises(ValueError, match="size does not match"):
        flw0.snap(idx0, mask=np.ones((2, 1)))


def test_downstream(flw0):
    idxs = np.arange(flw0.size, dtype=int)
    assert np.all(flw0.downstream(idxs).ravel()[flw0.mask] == flw0.idxs_ds[flw0.mask])
    with pytest.raises(ValueError, match="size does not match"):
        flw0.downstream(np.ones((2, 1)))


def test_sum_upstream(flw0):
    n_up = core.upstream_count(flw0.idxs_ds, flw0._mv)
    data = np.ones(flw0.shape, dtype=np.int32)
    assert np.all(flw0.upstream_sum(data).flat[flw0.mask] == n_up[flw0.mask])
    with pytest.raises(ValueError, match="size does not match"):
        flw0.upstream_sum(np.ones((2, 1)))


def test_moving_average(flw0, flwdir0_rank):
    idxs_seq = flwdir0_rank[2]
    data = np.random.random(flw0.shape)
    data_smooth = flw0.moving_average(data, n=1, weights=np.ones(flw0.shape))
    assert np.all(data_smooth == flw0.moving_average(data, n=1))
    idxs = flw0.path(idxs_seq[-1], max_length=2)[0][0]
    assert np.isclose(np.mean(data.flat[idxs]), data_smooth.flat[idxs[1]])
    with pytest.raises(ValueError, match="size does not match"):
        flw0.moving_average(np.ones((2, 1)), n=3)
    with pytest.raises(ValueError, match="size does not match"):
        flw0.moving_average(data, n=5, weights=np.ones((2, 1)))


def test_basins(flw0, flwdir0_rank):
    idxs_seq = flwdir0_rank[2]
    # basins
    basins = flw0.basins()
    assert basins.min() == 0
    assert basins.max() == flw0.idxs_pit.size
    assert basins.dtype == np.uint32
    assert np.all(basins.shape == flw0.shape)
    idx = np.arange(1, flw0.idxs_pit.size + 1, dtype=np.int16)
    assert flw0.basins(ids=idx).dtype == np.int16
    # subbasins
    subbasins = flw0.basins(idxs=idxs_seq[-4:])
    assert np.any(subbasins != basins)
    # errors
    with pytest.raises(ValueError, match="size does not match"):
        flw0.basins(ids=np.arange(flw0.idxs_pit.size - 1))
    with pytest.raises(ValueError, match="IDs cannot contain a value zero"):
        flw0.basins(ids=np.zeros(flw0.idxs_pit.size, dtype=np.int16))
    # basin bounds using IDENTITY transform
    lbs = flw0.basin_bounds(basins)[0]
    assert np.all(lbs == np.unique(basins[basins > 0]))
    lbs, _, total_bbox = flw0.basin_bounds(basins=np.ones(flw0.shape, dtype=np.uint32))
    assert np.all(np.abs(total_bbox[[1, 2]]) == flw0.shape)
    with pytest.raises(ValueError, match="shape does not match"):
        flw0.basin_bounds(basins=np.ones((2, 1)))
    # basin outlets
    idxs_out = flw0.basin_outlets(basins)[1]
    assert np.all(np.sort(idxs_out) == np.sort(flw0.idxs_pit))


def test_subbasins(flw0):
    pfaf = flw0.subbasins_pfafstetter()[0]
    bas0 = flw0.basins(flw0.idxs_pit[0])
    assert np.all(pfaf[bas0 != 0] > 0)
    assert pfaf.max() <= 9
    subbas = flw0.subbasins_streamorder()[0]
    assert np.all(subbas[bas0 != 0] > 0)
    subbas = flw0.subbasins_area(10)[0]
    assert np.all(subbas[bas0 != 0] > 0)


def test_uparea(flw0):
    # test with upstream grid cells
    uparea = flw0.upstream_area()
    assert uparea.min() == -9999
    assert uparea[uparea != -9999].min() == 1
    assert uparea.dtype == np.int32
    assert np.all(uparea.shape == flw0.shape)
    # compare with accuflux
    acc = flw0.accuflux(np.ones(flw0.shape))
    assert np.all(acc.flat[flw0.mask] == uparea.flat[flw0.mask])
    # test upstream area in km2
    uparea2 = flw0.upstream_area(unit="km2")
    assert uparea2.dtype == np.float32
    assert uparea2.max() == uparea2.flat[flw0.idxs_pit].max()
    with pytest.raises(ValueError, match="Unknown unit"):
        flw0.upstream_area(unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw0.accuflux(np.ones((2, 1)))
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw0.accuflux(np.ones((1, 1)), direction="???")


def test_streams(flw0, flwdir0_rank):
    idxs_seq = flwdir0_rank[2]
    # stream order
    strord = flw0.stream_order()
    assert strord.flat[flw0.mask].min() == 1
    assert strord.min() == 0
    assert strord.max() == strord.flat[flw0.idxs_pit].max() == 5
    assert strord.dtype == np.uint8
    assert np.all(strord.shape == flw0.shape)
    # stream segments
    feats = flw0.streams(strord=strord)
    fstrord = np.array([f["properties"]["strord"] for f in feats])
    findex = np.array([f["properties"]["idx"] for f in feats])
    assert np.all(fstrord == strord.flat[findex])
    findex_ds = np.array([f["properties"]["idx_ds"] for f in feats])
    # check agains Flwdir
    flw1 = pyflwdir.Flwdir(pyflwdir.flwdir.get_lin_indices(findex, findex_ds))
    assert np.all(fstrord == flw1.stream_order().ravel())
    # vectorize
    feats = flw0.vectorize()
    findex = np.array([f["properties"]["idx"] for f in feats])
    assert np.all(findex == np.sort(idxs_seq))
    with pytest.raises(ValueError, match="size does not match"):
        flw0.geofeatures([np.array([1, 2])], xs=np.arange(3), ys=np.arange(3))
    with pytest.raises(ValueError, match="size does not match"):
        flw0.streams(mask=np.ones((2, 1)))
    with pytest.raises(ValueError, match="Kwargs map"):
        flw0.geofeatures([np.array([1, 2])], uparea=np.ones((1, 1)))
    # stream distance
    data = np.zeros(flw0.shape, dtype=np.int32)
    data[flw0.rank > 0] = 1
    dist0 = flw0.accuflux(data, direction="down")
    assert dist0.dtype == np.int32
    dist = flw0.stream_distance(unit="cell")
    assert dist.max() == flw0.rank.max()
    assert dist.dtype == np.int32
    assert np.all(dist.shape == flw0.shape)
    assert np.all(dist0[dist != -9999] <= dist[dist != -9999])
    dist = flw0.stream_distance(mask=np.ones(flw0.shape, dtype=bool))
    assert np.all(dist[dist != -9999] == 0)
    dist = flw0.stream_distance(unit="m")
    assert dist.dtype == np.float32
    with pytest.raises(ValueError, match="Unknown unit"):
        flw0.stream_distance(unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw0.stream_distance(mask=np.ones((2, 1)))
    # river length
    data_smooth1 = flw0.smooth_rivlen(data, min_rivlen=0)
    assert np.all(data_smooth1 == data)


def test_upscale(flw0, nextxy0):
    flw1, idxs_out = flw0.upscale(5, method="dmm")  # single method
    assert flw1.transform[0] == 5 * flw0.transform[0]
    assert flw1.ftype == flw0.ftype
    flwerr = flw0.upscale_error(flw1, idxs_out)
    assert flwerr.flat[flw1.mask].min() == 0
    assert flwerr.flat[flw1.mask].max() == 1
    assert np.all(flwerr[flwerr < 0] == -1)
    with pytest.raises(ValueError, match="Unknown method"):
        flw0.upscale(5, method="unknown")
    with pytest.raises(ValueError, match="only works for D8 or LDD"):
        pyflwdir.from_array(nextxy0, ftype="nextxy").upscale(10)
    with pytest.raises(ValueError, match="size does not match"):
        flw0.upscale(5, uparea=np.ones((2, 1)))


def test_ucat(flw0):
    elevtn = flw0.rank
    hand = flw0.hand(elevtn=elevtn, drain=elevtn == 0)
    depths = np.linspace(0.5, 1, 2)
    idxs_out = flw0.ucat_outlets(5)
    ucat, ugrd = flw0.ucat_area(idxs_out)
    ucat1, uvol = flw0.ucat_volume(idxs_out, hand=hand, depths=depths)
    rivlen = flw0.subgrid_rivlen(idxs_out)
    rivslp = flw0.subgrid_rivslp(idxs_out, elevtn, length=1)
    rivwth = flw0.subgrid_rivavg(idxs_out, np.ones(flw0.shape))
    assert ugrd.shape == idxs_out.shape
    assert uvol.shape == (depths.size, *idxs_out.shape)
    assert ucat.shape == flw0.shape
    assert np.all(ucat1 == ucat)
    assert ugrd[idxs_out != flw0._mv].min() > 0
    assert ugrd[idxs_out != flw0._mv].min() > 0
    assert rivlen.shape == idxs_out.shape
    assert rivlen[idxs_out != flw0._mv].min() >= 0  # only zeros at boundary
    assert np.all(rivslp[idxs_out != flw0._mv] > 0)
    assert np.all(rivwth[idxs_out != flw0._mv] == 1)
    rivlen1 = flw0.subgrid_rivlen(idxs_out=None)
    assert rivlen1.shape == flw0.shape
    with pytest.raises(ValueError, match="Unknown method"):
        flw0.ucat_outlets(5, method="unkown")
    with pytest.raises(ValueError, match="Unknown unit"):
        flw0.ucat_area(idxs_out, unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw0.subgrid_rivslp(idxs_out, elevtn=np.ones((2, 1)))
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw0.subgrid_rivlen(idxs_out, direction="unknown")


def test_dem1():
    i = 867565
    rng = np.random.default_rng(i)
    dem = rng.random((15, 10), dtype=np.float32)
    flwdir = pyflwdir.from_dem(dem)
    dem1 = flwdir.dem_adjust(dem)
    assert np.all((dem1 - flwdir.downstream(dem1)) >= 0), i


def test_dem(flw0):
    elevtn = np.ones(flw0.shape)
    # create values that need fix
    diff = np.logical_and(flw0.rank == 2, flw0.upstream_sum(np.ones(flw0.shape)) >= 1)
    elevtn[diff] = 2.0
    elevtn_new = flw0.dem_adjust(elevtn)
    assert np.all(elevtn_new == 1.0)
    with pytest.raises(ValueError, match="size does not match"):
        flw0.dem_adjust(np.ones((2, 1)))
    # hand
    rank = flw0.rank
    drain = rank == 0
    hand = flw0.hand(drain, elevtn_new)
    assert np.all(hand[rank > 0] == 0)
    with pytest.raises(ValueError, match="size does not match"):
        flw0.hand(drain, np.ones((2, 1)))
    with pytest.raises(ValueError, match="size does not match"):
        flw0.hand(np.ones((2, 1)), elevtn_new)
    # floodplain
    fldpln = flw0.floodplains(elevtn_new, uparea=drain, upa_min=1, b=1)
    assert np.all(fldpln.flat[flw0.mask] == 1)
    with pytest.raises(ValueError, match="size does not match"):
        flw0.floodplains(np.ones((2, 1)))
