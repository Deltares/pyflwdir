# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module, specifically the wrapping of the methods which 
themselves are testes elsewhere"""

import pytest
import numpy as np
import pandas as pd
from affine import Affine

import pyflwdir
from pyflwdir import core, core_nextxy, core_d8

# test data
import test_core

idxs_ds = test_core.idxs_ds0
idxs_pit = test_core.idxs_pit0
idxs_seq = test_core.seq0
d8 = test_core.flwdir0
nextxy = core_nextxy.to_array(idxs_ds, d8.shape)
flw = pyflwdir.FlwdirRaster(idxs_ds.copy(), d8.shape, "d8", idxs_pit=idxs_pit.copy())
_mv = core._mv


@pytest.mark.parametrize("flwdir, ftype", [(d8, "d8"), (nextxy, "nextxy"),])
def test_from_to_array(flwdir, ftype):
    mask = np.ones(flwdir.shape)
    flw = pyflwdir.from_array(flwdir, mask=mask)
    assert flw.ftype == ftype
    assert np.all(pyflwdir.from_array(flw.to_array()).idxs_ds == flw.idxs_ds)
    with pytest.raises(ValueError, match="Invalid method"):
        flw.order_cells(method="???")


def test_from_array_errors():
    with pytest.raises(ValueError, match="could not be inferred."):
        pyflwdir.from_array(np.arange(20), ftype="infer")
    with pytest.raises(ValueError, match='ftype "unknown" unknown'):
        flw.to_array("unknown")
    with pytest.raises(ValueError, match="should be 2 dimensional"):
        pyflwdir.from_array(d8.ravel(), ftype="d8")
    with pytest.raises(ValueError, match="is not recognized."):
        pyflwdir.from_array(d8, ftype="ldd", check_ftype=True)
    with pytest.raises(ValueError, match="shape does not match"):
        pyflwdir.from_array(d8, mask=np.ones((1, 1)))


def test_flwdirraster_errors():
    with pytest.raises(ValueError, match="Unknown flow direction type"):
        pyflwdir.FlwdirRaster(idxs_ds, d8.shape, "unknown")
    with pytest.raises(ValueError, match="Invalid transform."):
        pyflwdir.FlwdirRaster(idxs_ds, d8.shape, "d8", transform=(0, 0))
    with pytest.raises(ValueError, match="Invalid FlwdirRaster: size"):
        pyflwdir.FlwdirRaster(idxs_ds[[0]], d8.shape, "d8")
    with pytest.raises(ValueError, match="Invalid FlwdirRaster: shape"):
        pyflwdir.FlwdirRaster(idxs_ds, (1, 2), "d8")
    with pytest.raises(ValueError, match="Invalid FlwdirRaster: no pits found"):
        pyflwdir.FlwdirRaster(np.array([1, 0], dtype=np.intp), (2, 1), "d8")


@pytest.mark.parametrize("parsed, d8", test_core.test_data)
def test_flwdirraster_attrs(parsed, d8):
    idxs_ds, idxs_pit, seq, rank = parsed
    flw = pyflwdir.FlwdirRaster(
        idxs_ds.copy(), d8.shape, "d8", idxs_pit=idxs_pit.copy()
    )
    assert flw.size == d8.size
    assert flw.shape == d8.shape
    assert isinstance(flw._dict, dict)
    assert isinstance(flw.__str__(), str)
    assert np.all(flw[flw.idxs_pit] == flw.idxs_pit)
    assert isinstance(flw.xy(flw.idxs_pit), tuple)
    assert isinstance(flw.transform, Affine)
    assert isinstance(flw.bounds, np.ndarray)
    assert isinstance(flw.latlon, bool)
    assert np.all(flw.rank.ravel() == rank)
    assert flw.ncells == seq.size
    assert np.all(np.diff(rank.flat[flw.idxs_seq]) >= 0)
    flw.repair_loops()
    assert flw.isvalid
    assert np.sum(flw.mask) == flw.ncells


def test_add_pits():
    idx0 = flw.idxs_pit
    x, y = flw.xy(flw.idxs_pit)
    # all cells are True -> pit at idx1
    flw.order_cells()  # set flw._seq
    flw.add_pits(idxs=idx0, streams=np.full(d8.shape, True, dtype=np.bool))
    assert np.all(flw.idxs_pit == idx0)
    assert flw._seq is None  # check if seq is deleted
    # original pit idx0
    flw.add_pits(xy=(x, y))
    assert np.all(flw.idxs_pit == idx0)
    # check some errors
    with pytest.raises(ValueError, match="size does not match"):
        flw.add_pits(idxs=idx0, streams=np.ones((1, 1)))
    with pytest.raises(ValueError, match="Either idxs or xy should be provided."):
        flw.add_pits()
    with pytest.raises(ValueError, match="Either idxs or xy should be provided."):
        flw.add_pits(idxs=idx0, xy=(x, y))


# NOTE tmpdir is predefined fixture
def test_save(tmpdir):
    fn = tmpdir.join("flw.pkl")
    flw.dump(fn)
    flw1 = pyflwdir.load(fn)
    for key in flw._dict:
        assert np.all(flw._dict[key] == flw1._dict[key])


def test_path_snap():
    idx0 = idxs_seq[-1]
    # up- & downstream
    path = flw.path(idx0)[0]
    idx1 = flw.snap(idx0)[0]
    assert np.all(flw.path(idx1, direction="up")[0][0][::-1] == path[0])
    assert np.all(flw.snap(idx1, direction="up")[0] == idx0)
    assert np.all(flw.snap(xy=flw.xy(idx1), direction="up")[0] == idx0)

    # with mask
    mask = np.full(flw.shape, False, dtype=np.bool)
    path, dist = flw.path(idx0, mask=mask)
    idx2, _ = flw.snap(idx0, mask=mask)
    assert path[0].size == dist[0] + 1
    assert idx1 == idx2[0] == path[0][-1]
    # no mask
    assert np.all(path[0] == flw.path(idx0)[0])
    assert np.all(idx1 == flw.snap(idx0)[0])
    # max dist
    l = int(np.round(dist[0] / 2))
    assert l <= flw.path(idx0, max_length=l)[1][0] <= dist[0]
    assert l <= flw.snap(idx0, max_length=l)[1][0] <= dist[0]
    with pytest.raises(ValueError, match="Unknown unit"):
        flw.path(idx0, unit="unknown")
    with pytest.raises(ValueError, match="Unknown unit"):
        flw.snap(idx0, unit="unknown")
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw.path(idx0, direction="unknown")
    with pytest.raises(ValueError, match="Unknown flow direction"):
        flw.snap(idx0, direction="unknown")
    with pytest.raises(ValueError, match="size does not match"):
        flw.path(idx0, mask=np.ones((1, 1)))
    with pytest.raises(ValueError, match="size does not match"):
        flw.snap(idx0, mask=np.ones((1, 1)))


def test_downstream():
    idxs = np.arange(flw.size, dtype=np.intp)
    assert np.all(flw.downstream(idxs).ravel()[flw.mask] == flw.idxs_ds[flw.mask])
    with pytest.raises(ValueError, match="size does not match"):
        flw.downstream(np.ones((1, 1)))


def test_sum_upstream():
    n_up = core.upstream_count(flw.idxs_ds)
    data = np.ones(flw.shape, dtype=np.int32)
    assert np.all(flw.upstream_sum(data).flat[flw.mask] == n_up[flw.mask])
    with pytest.raises(ValueError, match="size does not match"):
        flw.upstream_sum(np.ones((1, 1)))


def test_moving_average():
    data = np.random.random(flw.shape)
    data_smooth = flw.moving_average(data, n=1, weights=np.ones(flw.shape))
    assert np.all(data_smooth == flw.moving_average(data, n=1))
    idxs = flw.path(idxs_seq[-1], max_length=2)[0][0]
    assert np.isclose(np.mean(data.flat[idxs]), data_smooth.flat[idxs[1]])
    with pytest.raises(ValueError, match="size does not match"):
        flw.moving_average(np.ones((1, 1)), n=3)
    with pytest.raises(ValueError, match="size does not match"):
        flw.moving_average(data, n=5, weights=np.ones((1, 1)))


def test_basins():
    # basins
    basins = flw.basins()
    assert basins.min() == 0
    assert basins.max() == flw.idxs_pit.size
    assert basins.dtype == np.uint32
    assert np.all(basins.shape == flw.shape)
    idx = np.arange(1, flw.idxs_pit.size + 1, dtype=np.int16)
    assert flw.basins(ids=idx).dtype == np.int16
    # subbasins
    subbasins = flw.basins(idxs=idxs_seq[-4:])
    assert np.any(subbasins != basins)
    # errors
    with pytest.raises(ValueError, match="size does not match"):
        flw.basins(ids=np.arange(flw.idxs_pit.size - 1))
    with pytest.raises(ValueError, match="IDs cannot contain a value zero"):
        flw.basins(ids=np.zeros(flw.idxs_pit.size, dtype=np.int16))
    # basin bounds using IDENTITY transform
    df = flw.basin_bounds()
    assert isinstance(df, pd.DataFrame)
    assert np.all(df.index == np.unique(basins))  # including zero!
    df = flw.basin_bounds(basins=np.ones(flw.shape, dtype=np.uint32))
    assert np.all(df.loc[0, ["ymax", "xmax"]].values == flw.shape)
    with pytest.raises(ValueError, match="size does not match"):
        flw.basin_bounds(basins=np.ones((1, 1)))


def test_pfafstetter():
    pfaf = flw.pfafstetter(flw.idxs_pit[0])
    bas0 = flw.basins(flw.idxs_pit[0])
    assert np.all(pfaf[bas0 != 0] > 0)
    assert pfaf.max() <= 9


def test_uparea():
    # test with upstream grid cells
    uparea = flw.upstream_area()
    assert uparea.min() == -9999
    assert uparea[uparea != -9999].min() == 1
    assert uparea.dtype == np.int32
    assert np.all(uparea.shape == flw.shape)
    # compare with accuflux
    acc = flw.accuflux(np.ones(flw.shape))
    assert np.all(acc.flat[flw.mask] == uparea.flat[flw.mask])
    # test upstream area in km2
    uparea2 = flw.upstream_area(unit="km2")
    assert uparea2.dtype == np.float64
    assert uparea2.max() == uparea2.flat[flw.idxs_pit].max()
    with pytest.raises(ValueError, match="Unknown unit"):
        flw.upstream_area(unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw.accuflux(np.ones((1, 1)))


def test_streams():
    # stream order
    strord = flw.stream_order()
    assert strord.flat[flw.mask].min() == 1
    assert strord.min() == -1
    assert strord.max() == strord.flat[flw.idxs_pit].max()
    assert strord.dtype == np.int8
    assert np.all(strord.shape == flw.shape)
    # vectorize
    gdf = flw.vectorize()
    assert np.all(gdf.index == np.sort(idxs_seq))
    gdf["strord"] = strord.flat[gdf.index.values]
    gdf1 = flw.vectorize(mask=strord >= 3)
    assert np.all(gdf[gdf["strord"] >= 3].index == gdf1.index)
    with pytest.raises(ValueError, match="size does not match"):
        flw.vectorize(xs=np.arange(3), ys=np.arange(3))
    with pytest.raises(ValueError, match="size does not match"):
        flw.vectorize(mask=np.ones((1, 1)))
    # stream distance
    dist = flw.stream_distance(mask=np.ones(flw.shape, dtype=np.bool))
    assert dist.dtype == np.int32
    assert np.all(dist.shape == flw.shape)
    dist = flw.stream_distance(unit="m")
    assert dist.dtype == np.float64
    with pytest.raises(ValueError, match="Unknown unit"):
        flw.stream_distance(unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw.stream_distance(mask=np.ones((1, 1)))


def test_upscale():
    flw1, idxs_out = flw.upscale(5, method="dmm")
    assert flw1.transform[0] == 5 * flw.transform[0]
    assert flw1.ftype == flw.ftype
    subcon = flw.upscale_connect(flw1, idxs_out)
    assert subcon.flat[flw1.mask].min() == 0
    assert subcon.flat[flw1.mask].max() == 1
    assert np.all(subcon[subcon < 0] == -1)
    with pytest.raises(ValueError, match="Unknown method"):
        flw.upscale(5, method="unknown")
    with pytest.raises(ValueError, match="only works for D8 or LDD"):
        pyflwdir.from_array(nextxy, ftype="nextxy").upscale(10)
    with pytest.raises(ValueError, match="size does not match"):
        flw.upscale(5, uparea=np.ones((1, 1)))


def test_ucat():
    idxs_out = flw.ucat_outlets(5)
    ucat, ugrd = flw.ucat_area(idxs_out)
    assert ugrd.shape == idxs_out.shape
    assert ucat.shape == flw.shape
    assert ugrd[idxs_out != -1].min() > 0
    assert ugrd[idxs_out != -1].min() > 0
    rivlen, rivslp = flw.ucat_channel(idxs_out)
    assert rivlen.shape == idxs_out.shape
    assert rivlen[idxs_out != -1].min() >= 0  # only zeros at boundary
    assert np.all(rivslp[idxs_out != -1] == 0)
    rivlen, rivslp = flw.ucat_channel()
    assert rivlen.shape == flw.shape
    with pytest.raises(ValueError, match="Unknown method"):
        flw.ucat_outlets(5, method="unkown")
    with pytest.raises(ValueError, match="Unknown unit"):
        flw.ucat_area(idxs_out, unit="km")
    with pytest.raises(ValueError, match="size does not match"):
        flw.ucat_channel(idxs_out, elevtn=np.ones((1, 1)))


def test_dem():
    elevtn = np.ones(flw.shape)
    # create values that need fix
    diff = np.logical_and(flw.rank == 2, flw.upstream_sum(np.ones(flw.shape)) >= 1)
    elevtn[diff] = 2.0
    elevtn_new = flw.dem_adjust(elevtn)
    assert np.all(elevtn_new == 1.0)
    with pytest.raises(ValueError, match="size does not match"):
        flw.dem_adjust(np.ones((1, 1)))
    # hand
    rank = flw.rank
    drain = rank == 0
    hand = flw.hand(drain, elevtn_new)
    assert np.all(hand[rank > 0] == 0)
    with pytest.raises(ValueError, match="size does not match"):
        flw.hand(drain, np.ones((1, 1)))
    with pytest.raises(ValueError, match="size does not match"):
        flw.hand(np.ones((1, 1)), elevtn_new)
    # floodplain
    fldpln = flw.floodplains(elevtn_new, uparea=drain, upa_min=1, b=1)
    assert np.all(fldpln.flat[flw.mask] == 1)
    with pytest.raises(ValueError, match="size does not match"):
        flw.floodplains(np.ones((1, 1)))
