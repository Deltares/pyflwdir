# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
import numpy as np

import pyflwdir
from pyflwdir.core import _mv
from pyflwdir import dem


def test_dem_adjust():
    # # option 1 fill
    dem0 = np.array([8, 7, 6, 5, 5, 6, 5, 4])
    dem1 = np.array([8, 7, 6, 5, 5, 5, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # # option 2 fill
    dem0 = np.array([8, 7, 6, 5, 6, 6, 5, 4])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # option 3 dig and fill
    dem0 = np.array([8, 7, 6, 5, 6, 7, 5, 4])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # with dig on last position
    dem0 = np.array([8, 7, 6, 6, 6, 6, 5, 6])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 5])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
