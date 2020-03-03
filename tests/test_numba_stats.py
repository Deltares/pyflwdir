# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
import numpy as np

from pyflwdir import numba_stats as ns

def test_stats():
    data = np.random.rand(10)
    weights = np.random.rand(10)
    assert np.average(data, weights=weights) == ns._average(data, weights, np.nan)
    assert np.mean(data) == ns._mean(data, np.nan)
    data[-1] = np.nan
    assert np.average(data, weights=weights) != ns._average(data, weights, np.nan)
    data = np.random.randint(0,10,10)
    assert np.average(data, weights=weights) == ns._average(data, weights, np.nan)

    