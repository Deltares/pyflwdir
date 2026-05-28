import numpy as np
import pytest

from pyflwdir.flwdir import Flwdir, get_loc_idx


@pytest.fixture
def data():
    idx = np.array(
        [13924, 15144, 10043, 432, 7684, 6379, 6401, 3650, 2725, 95, 147, 7777]
    )
    # first idx_ds "15442" is not found in idx and interpreted as pit
    idx_ds = np.array(
        [15442, 13924, 13924, 10043, 10043, 7684, 7684, 6401, 6401, 2725, 2725, 7777]
    )
    # lin indices
    idxs_ds = np.array([0, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 11])
    # rank
    rank = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0])
    return idx, idx_ds, idxs_ds, rank


def test_get_loc_idx_int32(data):
    # unpack test data
    idx, idx_ds, idxs_ds, rank = data

    idx = idx.astype(np.int32)
    idx_ds = idx_ds.astype(np.int32)
    # test Flwdir (as in from_dataframe)
    _ = get_loc_idx(idx, idx_ds)


def test_njit():
    # from pyflwdir.flwdir import get_loc_idx
    import numpy as np
    from numba import njit

    @njit(cache=True)
    def get_loc_idx(idxs: np.ndarray, idxs_ds: np.ndarray) -> np.ndarray:
        """Get linear indices of downstream cells."""
        idx_map = {idx: i for i, idx in enumerate(idxs)}
        # return i if idx_ds not in idx_map, i.e. idx is a pit
        idxs_ds0 = np.empty_like(idxs, dtype=idxs.dtype)
        for i, idx_ds in enumerate(idxs_ds):
            idxs_ds0[i] = idx_map.get(idx_ds, i)
        return idxs_ds0

    idxs = np.array([97, 40, 6800, 3601, 6009, 8715], dtype=np.int32)
    idxs_ds = np.array([3601, 3601, 6009, 6009, 8715, 8715], dtype=np.int32)
    out = get_loc_idx(idxs=idxs, idxs_ds=idxs_ds)


def test_from_dataframe(data):
    # unpack test data
    idx, idx_ds, idxs_ds, rank = data

    # test Flwdir (as in from_dataframe)
    idxs_ds0 = get_loc_idx(idx, idx_ds)
    assert np.all(idxs_ds0 == idxs_ds)
    flwdir = Flwdir(idxs_ds=idxs_ds)
    assert np.all(flwdir.rank == rank)
    assert flwdir._mv == -1

    # test with uint64
    idxs_ds0 = get_loc_idx(idx.astype(np.uint64), idx_ds.astype(np.uint64))
    flwdir = Flwdir(idxs_ds=idxs_ds0)
    assert np.all(flwdir.rank == rank)
    assert flwdir._mv == 18446744073709551615
