#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the numba-sheds module.
"""
import pytest

from numba-sheds import numba-sheds


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_numba-sheds(an_object):
    assert an_object == {}
