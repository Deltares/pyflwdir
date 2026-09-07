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
    assert flwdir._mv == -1


def test_flwdir_uint64_indices_are_normalized(data):
    _, _, idxs_ds, rank = data
    idxs_pit = np.array([0, 11], dtype=np.uint64)
    idxs_seq = np.argsort(rank).astype(np.uint64)

    flwdir = Flwdir(
        idxs_ds=idxs_ds.astype(np.uint64),
        idxs_pit=idxs_pit,
        idxs_outlet=idxs_pit,
        idxs_seq=idxs_seq,
    )

    assert flwdir.idxs_ds.dtype == np.int64
    assert flwdir.idxs_pit.dtype == np.int64
    assert flwdir.idxs_outlet.dtype == np.int64
    assert flwdir.idxs_seq.dtype == np.int64
    assert flwdir._mv == -1
    assert np.all(flwdir.rank == rank)


def test_flwdir_uint64_indices_larger_than_int64_raise():
    idxs_ds = np.array([0, np.iinfo(np.int64).max + 1], dtype=np.uint64)

    with pytest.raises(ValueError, match="cannot be represented as int64"):
        Flwdir(idxs_ds=idxs_ds)
