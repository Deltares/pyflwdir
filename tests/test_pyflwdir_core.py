# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

"""Tests for the pyflwdir.core_xxx submodules.
"""

import pytest
import numpy as np

from pyflwdir import (
    core, core_d8, core_flow, core_ldd
)
from pyflwdir.core_conversion import ldd_to_d8, d8_to_ldd

_shape = (2, 678, 776)
_ldd = np.fromfile(r'./tests/data/ldd.bin', dtype=np.uint8).reshape(_shape[1:])
_d8 = np.fromfile(r'./tests/data/d8.bin', dtype=np.uint8).reshape(_shape[1:])
_nextxy = np.fromfile(r'./tests/data/nextxy.bin', dtype=np.int32).reshape(_shape)

def test_core_simple():
    for fd in [core_flow, core_d8, core_ldd]:
        _invalid = np.array([[2,4,8],[10,0,-1]], dtype=np.uint8)
        _shape = fd._us.shape
        if fd == core_flow:
            _invalid = np.stack([_invalid, _invalid])
            _shape = fd._us.shape[1:]
        # test isvalid
        assert fd.isvalid(fd._us), "isvalid test with _us data failed"
        assert not fd.isvalid(_invalid), "isvalid false test with invalid data failed"
        # test ispit
        assert np.all(fd.ispit(fd._pv)), "ispit test with _pv data failed"
        # test isnodata 
        assert np.all(fd.isnodata(fd._mv)), "isnodata test with _mv data failed"
        # test data type / dim errors
        with pytest.raises(TypeError):
            fd.from_flwdir(fd._us[None, ...])
            fd.from_flwdir(fd._us.astype(np.float))
        # test upstream / downstream / pit with _us data
        idxs_valid, idxs_ds, idxs_us, idx_pits = fd.from_flwdir(fd._us)
        assert np.all(idxs_valid == np.arange(9)), "valid idx test with _us data failed"
        assert np.all(idxs_ds == 4), "downstream idx test with _us data failed"
        assert (np.all(idx_pits == 4) and 
                idx_pits.size == 1), "pit idx test with _us data failed"
        assert (np.all(idxs_us[4,:]==np.array([0, 1, 2, 3, 5, 6, 7, 8])) and
                np.all(idxs_us[:4, :]==core._mv) and 
                np.all(idxs_us[5:, :]==core._mv)), "upstream idx test with _us data failed"
        assert np.all(fd.to_flwdir(idxs_valid, idxs_ds, _shape) == fd._us), "convert back with _us data failed"
        # test all invalid/pit with _ds data
        idxs_valid, idxs_ds, idxs_us, idx_pits = fd.from_flwdir(fd._ds)
        assert (np.all(idxs_valid == idx_pits) and 
                np.all(idxs_us == core._mv)), "test all pits with _ds data failed"

def test_core_realdata():
    _d8_ = core_d8.from_flwdir(_d8)
    fdict = dict(
        flow = core_flow.from_flwdir(_nextxy),
        ldd = core_ldd.from_flwdir(_ldd),
    )
    for ftype, _fd_ in fdict.items():
        for i in range(len(_d8_)):
            assert np.all(_d8_[i] == _fd_[i]), f"disagreement between 'd8' and '{ftype}'' output {i}"

def test_d8_dsus():
    """test d8_upstream and d8_upstream functions"""
    fd = core_d8
    _us_flat = fd._us.flatten()
    _ds_flat = fd._ds.flatten()
    shape = fd._us.shape
    # test upstream
    for idx0 in range(9):
        flwdir_flat = np.zeros(9, dtype=np.uint8)
        flwdir_flat[idx0] = np.uint8(1)
        flwdir_flat *= _us_flat
        if idx0 != 4:
            assert np.all(fd.us_indices(4, flwdir_flat, shape) == idx0)
        else:
            assert fd.us_indices(4, flwdir_flat, shape).size == 0
    # test downstream
    for idx0 in range(9):
        if idx0 != 4:
            assert fd.ds_index(idx0, _ds_flat, shape) == -1
        else:
            assert fd.ds_index(idx0, _ds_flat, shape) == 4
        assert fd.ds_index(idx0, _us_flat, shape, dd= _us_flat[idx0]) == 4
        assert fd.ds_index(idx0, _us_flat, shape) == 4
    # test idx_to_dd
    for idx0 in range(9):
        assert fd.idx_to_dd(idx0, 4, (3,3)) == _us_flat[idx0]


def test_conversion():
    assert np.all(d8_to_ldd(ldd_to_d8(core_ldd._ds)) == core_ldd._ds)
    assert np.all(ldd_to_d8(d8_to_ldd(core_d8._ds)) == core_d8._ds)


if __name__ == "__main__":
    test_core_simple()
    test_d8_dsus()
    test_conversion()
    test_core_realdata