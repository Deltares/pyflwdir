# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

"""Tests for the pyflwdir.core submodule.
"""

import pytest
import numpy as np
from pyflwdir import core
from pyflwdir.core import d8 as fd

def test_ds():
    _us_flat = fd._us.flatten()
    _ds_flat = fd._ds.flatten()
    shape = fd._us.shape
    # test downstream > ds_idx
    for idx0 in range(9):
        assert fd.ds_index(idx0, _us_flat, shape) == 4
    # test downstream > ds_idx out of range
    for idx0 in range(9):
        if idx0 != 4:
            assert fd.ds_index(0, _ds_flat, shape) == -1

def test_us():
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
    # test upstream > us_idxs out of range
    for idx0 in range(9):
        if idx0 != 4:
            assert fd.us_indices(0, _ds_flat, shape).size == 0

def test_main_us():
    _us_flat = fd._us.flatten()
    _ds_flat = fd._ds.flatten()
    shape = fd._us.shape
    # test upstream
    for idx0 in range(9):
        upa = np.zeros(9, dtype=np.float32)
        upa[idx0] = 1.
        if idx0 != 4:
            assert np.all(fd.us_main_indices(4, _us_flat, upa, shape, 0)[0] == idx0)
            assert fd.us_main_indices(4, _us_flat, upa, shape, 0)[1] == upa[idx0]
        else:
            assert fd.us_main_indices(4, _us_flat, upa, shape, 0)[0].size == 8 # all us are equally large
            assert fd.us_main_indices(4, _us_flat, upa, shape, 1)[0].size == 0 # non above threshold

                
def test_idx_to_d8():
    _us_flat = fd._us.flatten()
    _ds_flat = fd._ds.flatten()
    shape = fd._us.shape        
    # test idx
    for idx0 in range(9):
        assert fd.idx_to_dd(idx0, 4, (3,3)) == _us_flat[idx0]

if __name__ == "__main__":
    test_ds()
    test_us()
    test_idx_to_d8()
    test_main_us()
    print('success')
    pass