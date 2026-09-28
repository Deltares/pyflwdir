import os

import numpy as np
import pytest

from pyflwdir import core, core_d8, core_nextxy
from pyflwdir.pyflwdir import FlwdirRaster, from_dem


@pytest.fixture(scope="session")
def testdir():
    return os.path.dirname(__file__)


@pytest.fixture(scope="session")
def flwdir_real(testdir):
    return np.loadtxt(os.path.join(testdir, "data", "flwdir.asc"), dtype=np.uint8)


@pytest.fixture(scope="session")
def flwdir_real_idxs(flwdir_real):
    idxs_ds0, idxs_pit0, _ = core_d8.from_array(flwdir_real, dtype=np.uint32)
    return idxs_ds0, idxs_pit0


@pytest.fixture(scope="session")
def flwdir_real_rank(flwdir_real_idxs):
    idxs_ds0, _ = flwdir_real_idxs
    rank0, n0 = core.rank(idxs_ds0, mv=np.uint32(core._mv))
    seq0 = np.argsort(rank0)[-n0:]
    return rank0, n0, seq0


@pytest.fixture(scope="session")
def test_data_real(flwdir_real_idxs, flwdir_real_rank):
    rank0, _, seq0 = flwdir_real_rank
    idxs_ds0, idxs_pit0 = flwdir_real_idxs
    return idxs_ds0, idxs_pit0, seq0, rank0, np.uint32(core._mv)


@pytest.fixture(scope="session")
def nextxy_real(flwdir_real, flwdir_real_idxs):
    return core_nextxy.to_array(flwdir_real_idxs[0], flwdir_real.shape)


@pytest.fixture(scope="session")
def flw_real(flwdir_real, flwdir_real_idxs):
    idxs_ds0, idxs_pit0 = flwdir_real_idxs
    return FlwdirRaster(
        idxs_ds0.copy(), flwdir_real.shape, "d8", idxs_pit=idxs_pit0.copy(), cache=False
    )


@pytest.fixture(scope="session")
def flwdir_uint32():
    np.random.seed(2345)
    return from_dem(np.random.rand(15, 10)).to_array("d8")


@pytest.fixture(scope="session")
def flwdir_uint32_idxs(flwdir_uint32):
    idxs_ds1, idxs_pit1, _ = core_d8.from_array(flwdir_uint32, dtype=np.uint32)
    return idxs_ds1, idxs_pit1


@pytest.fixture(scope="session")
def flwdir_uint32_rank(flwdir_uint32_idxs):
    idxs_ds1, _ = flwdir_uint32_idxs
    rank1, n1 = core.rank(idxs_ds1, mv=np.uint32(core._mv))
    seq1 = np.argsort(rank1)[-n1:]
    return rank1, n1, seq1


@pytest.fixture(scope="session")
def test_data_uint32(flwdir_uint32_idxs, flwdir_uint32_rank):
    rank1, _, seq1 = flwdir_uint32_rank
    idxs_ds1, idxs_pit1 = flwdir_uint32_idxs
    return idxs_ds1, idxs_pit1, seq1, rank1, np.uint32(core._mv)


@pytest.fixture(scope="session")
def flwdir_int64():
    np.random.seed(2345)
    return from_dem(np.random.rand(15, 10)).to_array("d8")


@pytest.fixture(scope="session")
def flwdir_int64_idxs(flwdir_int64):
    idxs_ds2, idxs_pit2, _ = core_d8.from_array(flwdir_int64, dtype=np.int64)
    return idxs_ds2, idxs_pit2


@pytest.fixture(scope="session")
def flwdir_int64_rank(flwdir_int64_idxs):
    idxs_ds2, _ = flwdir_int64_idxs
    rank2, n2 = core.rank(idxs_ds2, mv=core._mv)
    seq2 = np.argsort(rank2)[-n2:]
    return rank2, n2, seq2


@pytest.fixture(scope="session")
def test_data_int64(flwdir_int64_idxs, flwdir_int64_rank):
    rank2, _, seq2 = flwdir_int64_rank
    idxs_ds2, idxs_pit2 = flwdir_int64_idxs
    return idxs_ds2, idxs_pit2, seq2, rank2, core._mv


# same flow directions as flwdir_real, with the int32 indices of a small raster
@pytest.fixture(scope="session")
def flwdir_real_int32(flwdir_real):
    return flwdir_real


@pytest.fixture(scope="session")
def flwdir_real_int32_idxs(flwdir_real_int32):
    idxs_ds3, idxs_pit3, _ = core_d8.from_array(flwdir_real_int32, dtype=np.int32)
    return idxs_ds3, idxs_pit3


@pytest.fixture(scope="session")
def flwdir_real_int32_rank(flwdir_real_int32_idxs):
    idxs_ds3, _ = flwdir_real_int32_idxs
    rank3, n3 = core.rank(idxs_ds3, mv=np.int32(core._mv))
    seq3 = np.argsort(rank3)[-n3:]
    return rank3, n3, seq3


@pytest.fixture(scope="session")
def test_data_real_int32(flwdir_real_int32_idxs, flwdir_real_int32_rank):
    rank3, _, seq3 = flwdir_real_int32_rank
    idxs_ds3, idxs_pit3 = flwdir_real_int32_idxs
    return idxs_ds3, idxs_pit3, seq3, rank3, np.int32(core._mv)


@pytest.fixture(scope="session")
def flw_real_int32(flwdir_real_int32, flwdir_real_int32_idxs):
    idxs_ds3, idxs_pit3 = flwdir_real_int32_idxs
    return FlwdirRaster(
        idxs_ds3.copy(),
        flwdir_real_int32.shape,
        "d8",
        idxs_pit=idxs_pit3.copy(),
        cache=False,
    )


@pytest.fixture(scope="session")
def flwdir_real_large(testdir):
    return np.loadtxt(os.path.join(testdir, "data", "flwdir1.asc"), dtype=np.uint8)


@pytest.fixture(scope="session")
def flwdir_real_large_idxs(flwdir_real_large):
    idxs_ds0, idxs_pit0, _ = core_d8.from_array(flwdir_real_large, dtype=np.uint32)
    return idxs_ds0, idxs_pit0


@pytest.fixture(scope="session")
def flwdir_real_large_idxs_int64(flwdir_real_large):
    idxs_ds0, idxs_pit0, _ = core_d8.from_array(flwdir_real_large, dtype=np.int64)
    return idxs_ds0, idxs_pit0
