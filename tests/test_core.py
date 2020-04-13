# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.core.py submodule."""

import pytest
import numpy as np

from pyflwdir import core, core_d8, ordered

flwdir_random = np.random.choice(core_d8._ds.flatten(), (10, 10))
parsed_random = core_d8.from_array(flwdir_random)

flwdir = np.array(
    [
        [1, 1, 1, 4, 4, 4, 8, 16, 2, 4],
        [1, 128, 128, 1, 4, 16, 1, 1, 1, 4],
        [1, 1, 1, 1, 4, 8, 1, 1, 4, 16],
        [128, 1, 128, 1, 1, 4, 1, 4, 16, 16],
        [1, 1, 1, 4, 1, 2, 1, 4, 16, 16],
        [1, 1, 1, 1, 1, 1, 1, 4, 8, 64],
        [1, 1, 2, 128, 64, 4, 2, 4, 64, 16],
        [1, 1, 2, 1, 1, 1, 1, 4, 16, 1],
        [247, 128, 1, 1, 1, 1, 2, 16, 16, 8],
        [128, 1, 1, 2, 1, 2, 4, 16, 16, 64],
    ],
    dtype=np.uint8,
)
parsed = core_d8.from_array(flwdir)


@pytest.mark.parametrize(
    "parsed, flwdir", [(parsed, flwdir), (parsed_random, flwdir_random)]
)
def test_downstream(parsed, flwdir):
    idxs_ds, idxs_pit, n = parsed
    ncol = flwdir.shape[1]
    # rank
    ranks, n1 = core.rank(idxs_ds)
    assert np.sum(ranks == 0) == idxs_pit.size
    if np.any(ranks > 0):
        idxs_mask = np.where(ranks > 0)[0]  # valid and no pits
        assert np.all(ranks[idxs_mask] == ranks[idxs_ds[idxs_mask]] + 1)
    # pit indices
    idxs_pit1 = np.sort(core.pit_indices(idxs_ds))
    assert np.all(idxs_pit1 == np.sort(idxs_pit))
    # loop indices
    idxs_loop = core.loop_indices(idxs_ds)
    assert np.all(ranks[idxs_loop] == -1)
    assert n1 == n - idxs_loop.size
    # local upstream indices
    if np.any(ranks >= 2):
        rmax = np.max(ranks)
        idxs = np.where(ranks == rmax)[0]
        # path
        paths, dists = core.path(idxs, idxs_ds, ncol)
        assert np.all([p.size for p in paths] == rmax + 1)
        assert np.all(dists == rmax)
        # snap
        idxs1, dists1 = core.snap(idxs, idxs_ds, ncol, real_length=True)
        assert np.all([idxs_ds[idx] == idx for idx in idxs1])
        assert np.all(dists1 >= rmax)
        idxs2, dists2 = core.snap(idxs, idxs_ds, ncol, real_length=False, max_length=2)
        assert np.all(dists2 == 2)
        assert np.all(ranks[idxs2] == rmax - 2)
        idxs2, dists2 = core.snap(idxs, idxs_ds, ncol, mask=ranks <= rmax - 2)
        assert np.all(dists2 == 2)
        assert np.all(ranks[idxs2] == rmax - 2)
        # window
        idx0 = np.where(ranks == 2)[0][0]
        path = core._trace(idx0, idxs_ds, ncol)[0]
        wdw = core._window(idx0, 2, idxs_ds, idxs_ds)
        assert np.all(path == wdw[2:]) and np.all(path[::-1] == wdw[:-2])


@pytest.mark.parametrize(
    "parsed, flwdir", [(parsed, flwdir), (parsed_random, flwdir_random)]
)
def test_upstream(parsed, flwdir):
    idxs_ds, idxs_pit, n = parsed
    ncol = flwdir.shape[1]
    seq = np.argsort(core.rank(idxs_ds)[0])[-n:]
    uparea = ordered.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    # count
    n_up = core.upstream_count(idxs_ds)
    assert np.sum(n_up[n_up != -9]) == n - idxs_pit.size
    # upstream matrix
    idxs_us = core.upstream_matrix(idxs_ds)
    assert np.sum(idxs_us != core._mv) == idxs_ds.size - idxs_pit.size
    # headwater
    idxs_headwater = core.headwater_indices(idxs_ds)
    assert np.sum(n_up == 0) == idxs_headwater.size
    if np.any(n_up > 0):
        # local upstream indices
        idx0 = np.where(uparea == np.max(uparea))[0][0]
        idxs_us0 = np.sort(core._upstream_d8_idx(idx0, idxs_ds, flwdir.shape))
        idxs_us1 = np.sort(idxs_us[idx0, : n_up[idx0]])
        assert np.all(idxs_us1 == idxs_us0)
        # main upstream
        idxs_us_main = core.main_upstream(idxs_ds, uparea)
        assert np.any(idxs_us0 == idxs_us_main[idx0])
        idxs = np.where(idxs_us_main != core._mv)[0]
        assert np.all(idxs_ds[idxs_us_main[idxs]] == idxs)
        assert idxs.size == n - idxs_headwater.size
        # tributary
        idxs_us_trib = core.main_tributary(idxs_ds, idxs_us_main, uparea)
        idxs = np.where(idxs_us_trib != core._mv)[0]
        assert idxs.size == np.sum(n_up > 1)
        if idxs.size > 0:
            assert np.all(idxs_ds[idxs_us_main[idxs]] == idxs)
        # window
        path = core._trace(idx0, idxs_us_main, ncol)[0]
        wdw = core._window(idx0, 1, idxs_us_main, idxs_us_main)
        assert np.all(path[:2] == wdw[1:]) and np.all(path[:2][::-1] == wdw[:-1])
        # tributaries
        idxs_trib = core._tributaries(idx0, idxs_us_main, idxs_us_trib, uparea)
        assert np.all(idxs_trib != idxs_us_trib[path])
        if idxs_trib.size > 1:
            idxs_trib1 = core._tributaries(
                idx0, idxs_us_main, idxs_us_trib, uparea, n=1
            )
            assert np.max(uparea[idxs_trib]) == uparea[idxs_trib1]