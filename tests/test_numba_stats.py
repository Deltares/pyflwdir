# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
import numpy as np

from pyflwdir import numba_stats as ns


def test_stats():
    nodata = -9999.0
    data = np.random.random(10)
    weights = np.random.random(10)
    assert np.isclose(
        np.average(data, weights=weights), ns._average(data, weights, nodata)
    )
    assert np.isclose(np.mean(data), ns._mean(data, nodata))
    data[-1] = nodata
    assert np.isclose(
        np.average(data[:-1], weights=weights[:-1]), ns._average(data, weights, nodata)
    )
    data = np.random.randint(0, 10, 10)
    assert np.isclose(
        np.average(data, weights=weights), ns._average(data, weights, nodata)
    )
