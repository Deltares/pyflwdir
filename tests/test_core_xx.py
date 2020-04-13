# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.core_xx.py and core_conversion submodules."""

import pytest
import numpy as np

from pyflwdir import core_d8, core_nextxy, core_ldd
from pyflwdir.core_conversion import ldd_to_d8, d8_to_ldd

# test data
from test_core import flwdir_random, parsed_random


@pytest.mark.parametrize("fd", [core_nextxy, core_d8, core_ldd])
def test_core(fd):
    """test core_x.py submodules based on _us definitions"""
    _name = fd._ftype
    # test isvalid
    assert fd.isvalid(fd._us)
    assert not fd.isvalid(fd._us * -1)
    # test ispit
    assert np.all(fd.ispit(fd._pv))
    # test isnodata
    assert np.all(fd.isnodata(fd._mv))
    # test from_array (and drdc)
    idxs_ds, idx_pits, n = fd.from_array(fd._us)
    assert n == 9
    assert np.all(idxs_ds == 4)
    assert np.all(idx_pits == 4) and idx_pits.size == 1
    # test to_array
    assert np.all(fd.to_array(idxs_ds, (3, 3)) == fd._us)


@pytest.mark.parametrize("fd", [core_d8, core_ldd])
def test_usds(fd):
    """assert D8 local upstream/ downstream operations"""
    _us_flat = fd._us.flatten()
    _ds_flat = fd._ds.flatten()
    shape = fd._us.shape
    # test upstream
    for idx0 in range(9):
        flwdir_flat = np.zeros(9, dtype=np.uint8)
        flwdir_flat[idx0] = np.uint8(1)
        flwdir_flat *= _us_flat
        if idx0 != 4:
            assert np.all(fd._upstream_idx(4, flwdir_flat, shape) == idx0)
        else:
            assert fd._upstream_idx(4, flwdir_flat, shape).size == 0
    # test downstream
    for idx0 in range(9):
        if idx0 != 4:
            assert fd._downstream_idx(idx0, _ds_flat, shape) == -1
        else:
            assert fd._downstream_idx(idx0, _ds_flat, shape) == 4
        assert fd._downstream_idx(idx0, _us_flat, shape) == 4


@pytest.mark.parametrize("fd", [core_nextxy, core_d8, core_ldd])
def test_identical(fd):
    """test if all core_xx.py return identical results"""
    idxs_ds0, idxs_pit0, n0 = parsed_random
    idxs_ds, idxs_pit, n = fd.from_array(fd.to_array(idxs_ds0, flwdir_random.shape))
    assert np.all(idxs_ds0 == idxs_ds)
    assert np.all(idxs_pit0 == idxs_pit)
    assert n0 == n


def test_ftype_conversion():
    """test conversion between d8 and ldd formats"""
    flwdir = np.random.choice(core_ldd._all, (10, 10))
    assert np.all(d8_to_ldd(ldd_to_d8(flwdir)) == flwdir)
