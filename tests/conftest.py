import os

import numpy as np
import pytest

# uncomment for debugging tests
os.environ["NUMBA_DISABLE_JIT"] = "1"

from pyflwdir import core, core_d8, core_nextxy  # noqa: E402
from pyflwdir.pyflwdir import FlwdirRaster, from_dem  # noqa: E402


@pytest.fixture(scope="session")
def testdir():
    return os.path.dirname(__file__)


@pytest.fixture(scope="session")
def flwdir0(testdir):
    return np.loadtxt(os.path.join(testdir, "data", "flwdir.asc"), dtype=np.uint8)


@pytest.fixture(scope="session")
def flwdir0_idxs(flwdir0):
    idxs_ds0, idxs_pit0, _ = core_d8.from_array(flwdir0, dtype=np.uint32)
    return idxs_ds0, idxs_pit0


@pytest.fixture(scope="session")
def flwdir0_rank(flwdir0_idxs):
    idxs_ds0, _ = flwdir0_idxs
    rank0, n0 = core.rank(idxs_ds0, mv=np.uint32(core._mv))
    seq0 = np.argsort(rank0)[-n0:]
    return rank0, n0, seq0


@pytest.fixture(scope="session")
def test_data0(flwdir0_idxs, flwdir0_rank):
    rank0, _, seq0 = flwdir0_rank
    idxs_ds0, idxs_pit0 = flwdir0_idxs
    return idxs_ds0, idxs_pit0, seq0, rank0, np.uint32(core._mv)


@pytest.fixture(scope="session")
def nextxy0(flwdir0, flwdir0_idxs):
    return core_nextxy.to_array(flwdir0_idxs[0], flwdir0.shape)


@pytest.fixture(scope="session")
def flw0(flwdir0, flwdir0_idxs):
    idxs_ds0, idxs_pit0 = flwdir0_idxs
    return FlwdirRaster(
        idxs_ds0.copy(), flwdir0.shape, "d8", idxs_pit=idxs_pit0.copy(), cache=False
    )


@pytest.fixture(scope="session")
def flwdir1():
    np.random.seed(2345)
    return from_dem(np.random.rand(15, 10)).to_array("d8")


@pytest.fixture(scope="session")
def flwdir1_idxs(flwdir1):
    idxs_ds1, idxs_pit1, _ = core_d8.from_array(flwdir1, dtype=np.uint32)
    return idxs_ds1, idxs_pit1


@pytest.fixture(scope="session")
def flwdir1_rank(flwdir1_idxs):
    idxs_ds1, _ = flwdir1_idxs
    rank1, n1 = core.rank(idxs_ds1, mv=np.uint32(core._mv))
    seq1 = np.argsort(rank1)[-n1:]
    return rank1, n1, seq1


@pytest.fixture(scope="session")
def test_data1(flwdir1_idxs, flwdir1_rank):
    rank1, _, seq1 = flwdir1_rank
    idxs_ds1, idxs_pit1 = flwdir1_idxs
    return idxs_ds1, idxs_pit1, seq1, rank1, np.uint32(core._mv)


@pytest.fixture(scope="session")
def test_data(test_data0, flwdir0, test_data1, flwdir1):
    return [(test_data0, flwdir0), (test_data1, flwdir1)]


@pytest.fixture(scope="session")
def flwdir_large(testdir):
    return np.loadtxt(os.path.join(testdir, "data", "flwdir1.asc"), dtype=np.uint8)


@pytest.fixture(scope="session")
def flwdir_large_idxs(flwdir_large):
    idxs_ds0, idxs_pit0, _ = core_d8.from_array(flwdir_large, dtype=np.uint32)
    return idxs_ds0, idxs_pit0
