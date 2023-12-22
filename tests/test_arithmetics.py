# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module."""

import numpy as np

from pyflwdir import arithmetics


def test_stats():
    nodata = -9999.0
    data = np.random.random(10)
    weights = np.random.random(10)
    assert np.isclose(
        np.average(data, weights=weights), arithmetics._average(data, weights, nodata)
    )
    assert np.isclose(np.mean(data), arithmetics._mean(data, nodata))
    data[-1] = nodata
    assert np.isclose(
        np.average(data[:-1], weights=weights[:-1]),
        arithmetics._average(data, weights, nodata),
    )
    assert np.isclose(np.average(data[:-1]), arithmetics._mean(data, nodata))
    data = np.random.randint(0, 10, 10)
    assert np.isclose(
        np.average(data, weights=weights), arithmetics._average(data, weights, nodata)
    )
