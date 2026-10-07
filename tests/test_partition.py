import numpy as np
import pytest
from multiprocessing import get_all_start_methods

import pyflwdir
from pyflwdir import partition, streams


def column_basins(nrow=8, ncol=8):
    """A raster whose columns are adjacent, independent drainage basins."""
    idxs_ds = np.arange(nrow * ncol, dtype=np.int32).reshape(nrow, ncol)
    idxs_ds[:-1] += ncol
    return pyflwdir.FlwdirRaster(
        idxs_ds.ravel(),
        shape=(nrow, ncol),
        ftype="d8",
    )


def tributaries_to_mainstem(nrow=15, ncol=15):
    """One basin with two equal tributaries joining every mainstem cell."""
    center = ncol // 2
    idxs_ds = np.arange(nrow * ncol, dtype=np.int32).reshape(nrow, ncol)
    idxs_ds[:, :center] += 1
    idxs_ds[:, center + 1 :] -= 1
    idxs_ds[:-1, center] += ncol
    return pyflwdir.FlwdirRaster(
        idxs_ds.ravel(),
        shape=(nrow, ncol),
        ftype="d8",
    )


@pytest.mark.unit
def test_basin_graph_uses_spatial_adjacency_and_cell_weights():
    flw = column_basins()
    basin_ids = flw.basins()
    labels, weights, rows, cols, adjacency = partition._basin_graph(
        basin_ids, flw.mask, flw.shape
    )

    assert labels.size == flw.shape[1]
    assert np.all(weights == flw.shape[0])
    assert np.allclose(rows, (flw.shape[0] - 1) / 2)
    assert np.allclose(cols, np.arange(flw.shape[1]))
    assert adjacency[0] == [1]
    assert adjacency[-1] == [flw.shape[1] - 2]
    assert all(len(neighbours) == 2 for neighbours in adjacency[1:-1])


@pytest.mark.unit
def test_metis_basin_partition_keeps_basins_whole_and_contiguous():
    pytest.importorskip("pymetis")
    flw = column_basins()
    plan = partition.partition_plan(flw, level="basin", refine=False)
    parts = plan.parts
    basin_ids = plan.basin_ids

    assert set(np.unique(parts)) == {0, 1, 2, 3}
    assert np.array_equal(plan.loads, np.full(4, flw.size // 4))
    for basin_id in np.unique(basin_ids):
        assert np.unique(parts[basin_ids == basin_id]).size == 1
    for part_id in range(4):
        columns = np.flatnonzero(np.any(parts == part_id, axis=0))
        assert np.all(np.diff(columns) == 1)


@pytest.mark.unit
def test_subbasin_plan_has_four_trunks_and_one_accounted_mainstem():
    flw = tributaries_to_mainstem()
    plan = partition.partition_plan(
        flw,
        level="subbasin",
        min_subtree_size=1,
    )
    parts = plan.parts.ravel()
    mask = flw.mask.ravel()

    assert set(np.unique(parts[mask])) == {0, 1, 2, 3, pyflwdir.MAINSTEM}
    assert plan.loads.sum() + plan.mainstem.size == flw.ncells
    assert plan.cut_outlets.size == plan.cut_inlets.size == plan.cut_ranks.size
    assert plan.mainstem.size > 0
    assert np.all(np.isin(plan.cut_inlets, plan.mainstem))
    if plan.predecessor >= 0:
        assert flw.idxs_ds[plan.predecessor] == plan.mainstem[0]

    cells = np.flatnonzero((parts >= 0) & (parts < partition.N_TRUNKS))
    downstream = flw.idxs_ds[cells]
    crossing = parts[downstream] != parts[cells]
    assert np.all(parts[downstream[crossing]] == pyflwdir.MAINSTEM)

    repeated = partition.partition_plan(
        flw,
        level="subbasin",
        min_subtree_size=1,
    )
    assert np.array_equal(repeated.parts, plan.parts)
    assert np.array_equal(repeated.cut_outlets, plan.cut_outlets)
    assert np.array_equal(repeated.mainstem, plan.mainstem)


@pytest.mark.unit
def test_empty_subbasin_ranks_receive_separated_spatial_anchors():
    records = [
        {
            "root": root,
            "position": root,
            "size": 9,
            "row": row,
            "col": col,
            "rank": -1,
        }
        for root, (row, col) in enumerate(
            [(10.0, 10.0), (10.0, 90.0), (90.0, 10.0), (90.0, 90.0)]
        )
    ]
    rank_rows = np.full(4, 50.0)
    rank_cols = np.full(4, 50.0)
    rank_occupied = np.array([True, False, False, False])
    cap = np.array([0, 25, 25, 25])

    partition._seed_empty_rank_centroids(
        records,
        rank_rows,
        rank_cols,
        rank_occupied,
        [1, 2, 3],
        cap,
        (100, 100),
        min_subtree_size=1,
    )

    anchors = set(zip(rank_rows[1:], rank_cols[1:]))
    assert len(anchors) == 3
    assert anchors <= {(10.0, 10.0), (10.0, 90.0), (90.0, 10.0), (90.0, 90.0)}


@pytest.mark.unit
def test_tributary_assignment_prefers_spatial_clusters():
    records = [
        {
            "root": root,
            "position": root,
            "size": 9,
            "row": row,
            "col": col,
            "rank": -1,
        }
        for root, (row, col) in enumerate(
            [
                (10.0, 10.0),
                (12.0, 12.0),
                (10.0, 50.0),
                (12.0, 52.0),
                (10.0, 90.0),
                (12.0, 88.0),
            ]
        )
    ]
    load = np.array([60, 0, 0, 0])
    rank_rows = np.array([50.0, 10.0, 10.0, 10.0])
    rank_cols = np.array([50.0, 10.0, 50.0, 90.0])

    partition._assign_tributary_records(
        records,
        load,
        rank_rows,
        rank_cols,
        max_rank=0,
        target=15,
        shape=(100, 100),
        min_subtree_size=1,
        imbalance_target=1.3,
    )

    assert records[0]["rank"] == records[1]["rank"] == 1
    assert records[2]["rank"] == records[3]["rank"] == 2
    assert records[4]["rank"] == records[5]["rank"] == 3


@pytest.mark.unit
def test_four_trunks_and_mainstem_walk_match_serial_accumulation():
    flw = tributaries_to_mainstem()
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    serial = flw.accuflux(data)
    plan = partition.partition_plan(
        flw,
        level="subbasin",
        min_subtree_size=1,
    )
    parts = plan.parts.ravel()
    cells, offsets = flw._layer_decomposition("cfds")
    accu = data.copy()

    for part_id in range(partition.N_TRUNKS):
        selected = parts[cells] == part_id
        part_cells = np.ascontiguousarray(cells[selected])
        layer_sizes = np.array(
            [
                np.count_nonzero(
                    parts[cells[offsets[layer] : offsets[layer + 1]]] == part_id
                )
                for layer in range(offsets.size - 1)
            ],
            dtype=np.int64,
        )
        part_offsets = np.zeros(offsets.size, dtype=np.int64)
        np.cumsum(layer_sizes, out=part_offsets[1:])
        streams.accuflux_partitioned_push(
            flw.idxs_ds,
            parts,
            part_id,
            part_cells,
            part_offsets,
            accu,
            -9999,
        )

    streams.accuflux_subbasin_mainstem(
        plan.mainstem,
        plan.predecessor,
        plan.cut_outlets,
        plan.cut_inlets,
        data,
        accu,
        -9999,
    )
    assert np.array_equal(accu, serial)


@pytest.mark.integration
def test_metis_process_and_thread_accumulation_matches_serial():
    pytest.importorskip("pymetis")
    flw = column_basins(nrow=64, ncol=8)
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
    hybrid = flw.accuflux(
        data,
        parallel=True,
        n_processes=4,
        threads_per_process=1,
        partition_level="basin",
        start_method=start_method,
    )
    assert np.array_equal(hybrid, flw.accuflux(data))


@pytest.mark.integration
def test_subbasin_process_path_uses_requested_threshold_and_matches_serial():
    flw = tributaries_to_mainstem(nrow=31, ncol=31)
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
    hybrid = flw.accuflux(
        data,
        parallel=True,
        n_processes=4,
        threads_per_process=1,
        partition_level="subbasin",
        partition_min_subtree_size=1,
        start_method=start_method,
    )
    assert np.array_equal(hybrid, flw.accuflux(data))
    plan, _ = pyflwdir.parallel._plan(flw, 4, "subbasin", 1, 1.05)
    assert plan.mainstem.size > 0
    assert set(np.unique(plan.parts)) == {0, 1, 2, 3, pyflwdir.MAINSTEM}


@pytest.mark.unit
def test_subbasin_partition_requires_the_flowtopo_four_trunks():
    flw = column_basins()
    with pytest.raises(ValueError, match="requires n_parts=4"):
        partition.partition_plan(flw, level="subbasin", n_parts=3)


@pytest.mark.unit
def test_metis_partition_requires_a_raster():
    flw = pyflwdir.Flwdir(np.array([0, 0], dtype=np.int32))
    with pytest.raises(TypeError, match="FlwdirRaster"):
        partition.partition_plan(flw, level="basin")


@pytest.mark.unit
def test_one_part_partition_is_trivial_without_metis():
    flw = column_basins()
    parts, load = flw.partition(level="basin", n_parts=1)
    assert np.all(parts == 0)
    assert np.array_equal(load, np.array([flw.ncells]))
