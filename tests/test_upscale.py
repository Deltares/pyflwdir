# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.upscael module."""

import pytest
from affine import Affine
import time
import numpy as np

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
parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, _, mv = [p.copy() for p in parsed]
cellsize = 5
tests = [("dmm", 7), ("eam", 3), ("com", 1), ("com2", 0)]

# caculate upstream area and basin
upa = streams.upstream_area(idxs_ds, seq, flwdir.shape[1], dtype=np.int32)
ids = np.arange(1, idxs_pit.size + 1, dtype=np.int)
bas = basins.basins(idxs_ds, idxs_pit, seq, ids)

# configure tests with different upscale methods
@pytest.mark.parametrize("name, discon", tests)
def test_upscale(name, discon):
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
    connect = upscale.connected(idxs_out, idxs_ds1, idxs_ds, mv=mv)
    assert np.sum(connect == 0) == discon


# TODO: extend tests
def test_map():
    upscale.map_celledge(idxs_ds, flwdir.shape, cellsize, mv=mv)
    upscale.map_effare(idxs_ds, flwdir.shape, cellsize, mv=mv)
