# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
from affine import Affine
import time
import numpy as np

# local
from pyflwdir import upscale, core_d8, core, streams, basins

# large test data
# import rasterio
# with rasterio.open(r'./data/ireland_dir.tif') as src:
#     flwdir = src.read(1)
# tests = [("dmm", 52160), ("eam", 18484), ("com", 5422), ("com2", 1427)]
# flwdir = np.fromfile(r"./data/d8.bin", dtype=np.uint8).reshape((678, 776))
# tests = [("dmm", 1073), ("eam", 406), ("com", 138), ("com2", 54)]
# idxs_ds, idxs_pit, _ = core_d8.from_array(flwdir)
# ranks, n = core.rank(idxs_ds)
# seq = np.argsort(ranks)[-n:]
# cellsize = 10

# small test data
from test_core import test_data

(idxs_ds, idxs_pit, seq, ranks), flwdir = test_data[0]
cellsize = 5
tests = [("dmm", 7), ("eam", 5), ("com", 3), ("com2", 2)]

# caculate upstream area and basin
upa = streams.upstream_area(idxs_ds, seq, flwdir.shape[1], dtype=np.int32)
ids = np.arange(1, idxs_pit.size + 1, dtype=np.int)
bas = basins.basins(idxs_ds, idxs_pit, seq, ids)

# configure tests with different upscale methods
@pytest.mark.parametrize("name, discon", tests)
def test_upscale(name, discon):
    fupscale = getattr(upscale, name)
    idxs_ds1, idxs_out = fupscale(idxs_ds, upa, flwdir.shape, cellsize)
    assert core.loop_indices(idxs_ds1).size == 0
    pit_idxs = core.pit_indices(idxs_ds1)
    assert np.unique(idxs_out[pit_idxs]).size == pit_idxs.size
    pit_bas = bas[idxs_out[pit_idxs]]
    assert np.unique(pit_bas).size == pit_bas.size
    # check number of disconnected cells for each method
    connect = upscale.connected(idxs_out, idxs_ds1, idxs_ds)
    assert np.sum(connect == 0) == discon
