"""Tests for the pyflwdir.upscale module."""

import numpy as np
import pytest

# local
from pyflwdir import basins, core, streams, upscale

# # large test data
# from pyflwdir import core_d8
# flwdir = np.fromfile(r"./data/d8.bin", dtype=np.uint8).reshape((678, 776))
# tests = [("dmm", 1073), ("eam", 406), ("com", 138), ("com2", 54)]
# idxs_ds, idxs_pit, _ = core_d8.from_array(flwdir)
# rank, n = core.rank(idxs_ds)
# seq = np.argsort(rank)[-n:]
# cellsize = 10

# cellsize = 20
tests = [
    (20, "dmm", 33),
    (20, "eam", 4),
    (20, "eam_plus", 2),
    (40, "ihu", 0),
    (20, "ihu", 1),
    (10, "ihu", 4),
    (5, "ihu", 7),
]


# configure tests with different upscale methods
@pytest.mark.integration
@pytest.mark.parametrize("cellsize, name, nflwerr", tests)
@pytest.mark.parametrize(
    "idxs", ["flwdir_real_large_idxs", "flwdir_real_large_idxs_int64"]
)
def test_upscale(cellsize, name, nflwerr, idxs, flwdir_real_large, request):
    flwdir = flwdir_real_large
    idxs_ds, idxs_pit = request.getfixturevalue(idxs)
    mv = idxs_ds.dtype.type(core._mv)
    # caculate upstream area and basin
    rank, n = core.rank(idxs_ds, mv=mv)
    seq = np.argsort(rank)[-n:]
    upa = streams.upstream_area(idxs_ds, seq, flwdir.shape[1], dtype=np.int32)
    ids = np.arange(1, idxs_pit.size + 1, dtype=int)
    bas = basins.basins(idxs_ds, idxs_pit, seq, ids)
    # upscale
    fupscale = getattr(upscale, name)
    idxs_ds1, idxs_out, shape1 = fupscale(idxs_ds, upa, flwdir.shape, cellsize, mv=mv)
    assert np.multiply(*shape1) == idxs_ds1.size
    assert idxs_ds.dtype == idxs_ds1.dtype
    assert core.loop_indices(idxs_ds1, mv=mv).size == 0
    pit_idxs = core.pit_indices(idxs_ds1)
    assert np.unique(idxs_out[pit_idxs]).size == pit_idxs.size
    pit_bas = bas[idxs_out[pit_idxs]]
    assert np.unique(pit_bas).size == pit_bas.size
    # check number of disconnected cells for each method
    flwerr_idxs = upscale.upscale_error(idxs_out, idxs_ds1, idxs_ds, mv=mv)[1]
    assert flwerr_idxs.size == nflwerr


# TODO: extend tests
@pytest.mark.integration
@pytest.mark.parametrize(
    "idxs", ["flwdir_real_large_idxs", "flwdir_real_large_idxs_int64"]
)
def test_map(idxs, flwdir_real_large, request):
    idxs_ds = request.getfixturevalue(idxs)[0]
    mv = idxs_ds.dtype.type(core._mv)
    upscale.map_celledge(idxs_ds, flwdir_real_large.shape, 20, mv=mv)
    upscale.map_effare(idxs_ds, flwdir_real_large.shape, 20, mv=mv)
