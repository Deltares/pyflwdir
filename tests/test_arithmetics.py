"""Tests for the pyflwdir module."""

import numpy as np
import pytest

from pyflwdir import arithmetics


@pytest.mark.unit
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


@pytest.mark.unit
def test_upstream_sum_nodata():
    nodata = -9999.0
    # 0 -> 1 -> 2 -> 3 (pit); 4 -> 3
    idxs_ds = np.array([1, 2, 3, 3, 3])
    data = np.array([1.0, nodata, 3.0, 4.0, 5.0])
    out = arithmetics.upstream_sum(idxs_ds, data, nodata=nodata)
    # cell 0 has no upstream cells; cells 1 and 2 drain from / into a missing value
    assert np.array_equal(out, [0.0, nodata, nodata, 8.0, 0.0])
    # the flag is kept if a valid upstream value is visited later
    idxs_ds = np.array([2, 2, 2])
    data = np.array([nodata, 5.0, 1.0])
    out = arithmetics.upstream_sum(idxs_ds, data, nodata=nodata)
    assert np.array_equal(out, [0.0, 0.0, nodata])
