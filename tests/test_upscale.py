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
from pyflwdir import (core, core_conversion, gis_utils, FlwdirRaster, upscale,
                      subgrid)
_mv = core._mv


@pytest.fixture
def d8_data():
    data = np.fromfile(r'./tests/data/d8.bin', dtype=np.uint8)
    return data.reshape((678, 776))


@pytest.fixture
def d8(d8_data):
    return FlwdirRaster(d8_data, ftype='d8', check_ftype=False)


# configure tests with different options
# n is number of disconnected cells per method
tests = {
    'dmm': {
        'n': 1073
    },
    'eam': {
        'n': 406
    },
    'cosm': {
        'kwargs': {
            'iter2': False
        },
        'n': 138
    },
    'cosm2': {
        'method': 'cosm',
        'n': 67
    },
}


def test_upscale(d8):
    cellsize = 10
    uparea = d8.upstream_area(latlon=False).ravel()
    basins = d8.basins().ravel()
    for name, mdict in tests.items():
        fupscale = getattr(upscale, mdict.get('method', name))
        kwargs = mdict.get('kwargs', {})
        nextidx, subidxs_out = fupscale(d8._idxs_ds, d8._idxs_valid, uparea,
                                        d8.shape, cellsize, **kwargs)
        d8_lr = FlwdirRaster(nextidx, ftype='nextidx', check_ftype=True)
        subidxs_out = subidxs_out.ravel()
        assert d8_lr.isvalid
        # check if in d8
        try:
            d8_lr.to_array(ftype='d8')
        except:
            pytest.fail(f'{name} not connected in d8')
        pit_idxs = nextidx.flat[d8_lr.pits]
        assert np.unique(pit_idxs).size == pit_idxs.size, name
        pitbas = basins[subidxs_out[d8_lr.pits]]
        assert np.unique(pitbas).size == pitbas.size, name
        # check number of disconnected cells for each method
        subidxs_out0 = d8._internal_idx(subidxs_out[d8_lr._idxs_valid])
        connect = subgrid.connected(subidxs_out0, d8_lr._idxs_ds, d8._idxs_ds)
        assert np.sum(~connect) == mdict['n'], name