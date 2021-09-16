# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.upscael module."""

import pytest
from affine import Affine
import time
import numpy as np
import os

# local
from pyflwdir import upscale, core_d8, core, streams, basins
from test_core import test_data

# # large test data
# flwdir = np.fromfile(r"./data/d8.bin", dtype=np.uint8).reshape((678, 776))
# tests = [("dmm", 1073), ("eam", 406), ("com", 138), ("com2", 54)]
# idxs_ds, idxs_pit, _ = core_d8.from_array(flwdir)
# rank, n = core.rank(idxs_ds)
# seq = np.argsort(rank)[-n:]
# cellsize = 10

# small test data
mv = np.uint32(core._mv)
# parsed0, flwdir0 = test_data[0]
# idxs_ds0, idxs_pit0 = [p.copy() for p in parsed0[:2]]
testdir = os.path.dirname(__file__)
flwdir = np.loadtxt(os.path.join(testdir, "flwdir1.asc"), dtype=np.uint8)
idxs_ds, idxs_pit, _ = core_d8.from_array(flwdir, dtype=np.uint32)

# cellsize = 20
tests = [
    (flwdir, idxs_ds, idxs_pit, 20, "dmm", 33),
    (flwdir, idxs_ds, idxs_pit, 20, "eam", 4),
    (flwdir, idxs_ds, idxs_pit, 20, "eam_plus", 2),
    (flwdir, idxs_ds, idxs_pit, 40, "ihu", 0),
    (flwdir, idxs_ds, idxs_pit, 20, "ihu", 1),
    (flwdir, idxs_ds, idxs_pit, 10, "ihu", 4),
    (flwdir, idxs_ds, idxs_pit, 5, "ihu", 7),
]

# configure tests with different upscale methods
@pytest.mark.parametrize("flwdir, idxs_ds, idxs_pit, cellsize, name, nflwerr", tests)
def test_upscale(flwdir, idxs_ds, idxs_pit, cellsize, name, nflwerr):
    # caculate upstream area and basin
    rank, n = core.rank(idxs_ds, mv=np.uint32(mv))
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
def test_map():
    upscale.map_celledge(idxs_ds, flwdir.shape, 20, mv=mv)
    upscale.map_effare(idxs_ds, flwdir.shape, 20, mv=mv)
