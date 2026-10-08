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


def rivers(nrow, widths):
    """Side by side basins; in each, rows drain sideways into a mainstem column
    that drains south to a pit on the last row."""
    ncol = int(sum(widths))
    cells = np.arange(nrow * ncol, dtype=np.int32).reshape(nrow, ncol)
    idxs_ds = cells.copy()
    first = 0
    for width in widths:
        center = first + width // 2
        idxs_ds[:, first:center] = cells[:, first:center] + 1
        idxs_ds[:, center + 1 : first + width] = cells[:, center + 1 : first + width] - 1
        idxs_ds[:-1, center] = cells[:-1, center] + ncol
        first += width
    return pyflwdir.FlwdirRaster(idxs_ds.ravel(), shape=(nrow, ncol), ftype="d8")


def tributaries_to_mainstem(nrow=15, ncol=15):
    """One basin with two equal tributaries joining every mainstem cell."""
    return rivers(nrow, [ncol])


def line_graph(weights):
    weights = np.asarray(weights, dtype=np.int64)
    n = weights.size
    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    return partition.PartitionGraph(
        weights,
        np.zeros(n),
        np.arange(n, dtype=np.float64),
        edges,
        np.ones(n - 1, dtype=np.int64),
    )


def accumulate_by_plan(flw, plan, data):
    """The four parts with CFDS threads' kernel, then every fifth region."""
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
            flw.idxs_ds, parts, part_id, part_cells, part_offsets, accu, -9999
        )
    for stem in plan.stems:
        streams.accuflux_subbasin_mainstem(
            stem.mainstem,
            stem.predecessor,
            stem.cut_outlets,
            stem.cut_inlets,
            data,
            accu,
            -9999,
        )
    return accu


def ground_pieces(flw, parts, part):
    """Pieces of one part on the ground, the two banks of a fifth region joined."""
    from scipy import ndimage

    grid = parts.reshape(flw.shape)
    labels, count = ndimage.label(
        (grid == part) | (grid == pyflwdir.MAINSTEM), structure=np.ones((3, 3))
    )
    return len(set(labels[grid == part].tolist()))


# --- the graph -----------------------------------------------------------------


@pytest.mark.unit
def test_graph_counts_cells_and_shared_sides():
    flw = column_basins()
    nodes = (flw.basins() - 1).astype(np.int64)
    graph = partition.graph_from_node_raster(nodes, flw.shape[1])
    assert np.all(graph.weights == flw.shape[0])
    assert np.allclose(graph.rows, (flw.shape[0] - 1) / 2)
    assert np.allclose(graph.cols, np.arange(flw.shape[1]))
    assert np.array_equal(
        graph.edges, np.column_stack((np.arange(7), np.arange(1, 8)))
    )
    assert np.all(graph.edge_weights == flw.shape[0])


@pytest.mark.unit
def test_merge_edges_sums_and_drops_loops():
    edges, weights = partition.merge_edges(
        [np.array([[1, 0], [0, 1], [2, 2]]), np.array([[0, 1]])],
        3,
        [np.array([2, 3, 9]), np.array([4])],
    )
    assert np.array_equal(edges, [[0, 1]])
    assert np.array_equal(weights, [9])


@pytest.mark.unit
def test_apportion_gives_every_land_a_part_in_proportion():
    assert np.array_equal(partition._apportion(np.array([80.0, 40.0]), 3), [2, 1])
    assert np.array_equal(partition._apportion(np.array([10.0, 10.0, 1.0]), 3), [1, 1, 1])
    assert partition._apportion(np.array([5.0]), 4).sum() == 4


# --- the balance refinement ----------------------------------------------------


def peninsula_graph():
    """part 1 = {p}; part 0 = {c, a, b, x, y, z} with c holding the peninsula
    a-b and touching the body x-y-z and p."""
    names = ["p", "c", "a", "b", "x", "y", "z"]
    weights = np.array([1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    pairs = [("p", "c"), ("c", "a"), ("a", "b"), ("c", "x"), ("x", "y"), ("y", "z")]
    edges = np.array([sorted((names.index(a), names.index(b))) for a, b in pairs])
    graph = partition.PartitionGraph(
        weights,
        np.zeros(7),
        np.arange(7, dtype=np.float64),
        edges,
        np.ones(len(pairs), dtype=np.int64),
    )
    parts = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    return names, graph, parts


@pytest.mark.unit
def test_detach_takes_the_peninsula_and_leaves_the_body():
    names, graph, parts = peninsula_graph()
    xadj, adjncy, _ = partition._csr(graph.size, graph.edges, graph.edge_weights)
    out = np.empty(graph.size, dtype=np.int64)
    taken, _ = partition._detach(
        xadj,
        adjncy,
        graph.weights,
        parts,
        names.index("c"),
        0,
        np.zeros(graph.size, dtype=np.int64),
        np.empty(graph.size, dtype=np.int64),
        1,
        100,
        out,
    )
    assert sorted(names[node] for node in out[:taken]) == ["a", "b", "c"]


@pytest.mark.unit
def test_refinement_moves_peninsulas_and_keeps_parts_connected():
    names, graph, parts = peninsula_graph()
    moves = partition._refine(graph, parts, 2, np.array([0.5, 0.5]), 1.0)
    assert moves == 1
    assert {names[node] for node in np.flatnonzero(parts == 1)} == {"p", "c", "a", "b"}
    assert {names[node] for node in np.flatnonzero(parts == 0)} == {"x", "y", "z"}


@pytest.mark.unit
def test_refinement_balances_a_line():
    graph = line_graph(np.ones(10))
    parts = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    partition._refine(graph, parts, 2, np.array([0.5, 0.5]), 1.0)
    assert np.array_equal(parts, [0] * 5 + [1] * 5)


@pytest.mark.unit
def test_fragments_join_the_part_around_them():
    # part 0 = 0..4 and a one-node fragment 7 inside part 1 = 5..9; island 10 stays
    graph = line_graph(np.ones(10))
    graph = partition.PartitionGraph(
        np.r_[graph.weights, 1],
        np.zeros(11),
        np.arange(11, dtype=np.float64),
        graph.edges,
        graph.edge_weights,
    )
    parts = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int32)
    assert partition._absorb_fragments(graph, parts, 2) == 1
    assert np.array_equal(parts, [0] * 5 + [1] * 5 + [0])


# --- Method 1 and Method 2 on graphs --------------------------------------------


@pytest.mark.unit
def test_dominant_basin_is_a_part_of_its_own():
    pytest.importorskip("pymetis")
    graph = line_graph([70] + [10] * 9)
    parts = partition.assign_basins(graph, 4, imbalance_target=1.0)
    assert parts[0] == 0 and np.all(parts[1:] != 0)
    loads = np.bincount(parts, weights=graph.weights, minlength=4)
    assert np.array_equal(loads, [70, 30, 30, 30])
    for part in (1, 2, 3):
        assert np.all(np.diff(np.flatnonzero(parts == part)) == 1)


@pytest.mark.unit
def test_basins_a_dominant_basin_cuts_off_stay_with_it():
    pytest.importorskip("pymetis")
    # node 0 is dominant; node 1 touches only node 0 (an enclave); 2..9 a line
    weights = np.array([70, 2] + [10] * 8, dtype=np.int64)
    edges = np.array([[0, 1], [0, 2]] + [[i, i + 1] for i in range(2, 9)])
    rows = np.zeros(10)
    cols = np.array([0, -5] + list(range(1, 9)), dtype=np.float64)
    graph = partition.PartitionGraph(
        weights, rows, cols, edges, np.ones(edges.shape[0], dtype=np.int64)
    )
    parts = partition.assign_basins(graph, 4, imbalance_target=1.0)
    assert parts[1] == parts[0]
    assert np.all(parts[2:] != parts[0])


@pytest.mark.unit
def test_parts_never_span_land_that_does_not_touch():
    pytest.importorskip("pymetis")
    # land A: nodes 0..7, land B: nodes 8..11 (apart), an island 12 by B
    weights = np.array([10] * 12 + [1], dtype=np.int64)
    edges = np.array([[i, i + 1] for i in range(7)] + [[i, i + 1] for i in range(8, 11)])
    rows = np.zeros(13)
    cols = np.array(list(range(8)) + [100, 101, 102, 103, 104.5], dtype=np.float64)
    graph = partition.PartitionGraph(
        weights, rows, cols, edges, np.ones(edges.shape[0], dtype=np.int64)
    )
    parts = partition._partition_components(graph, graph, 3, 42, 1.0, True)
    assert set(parts[:8]).isdisjoint(set(parts[8:12]))
    assert len(set(parts[:8])) == 2 and len(set(parts[8:12])) == 1
    assert parts[12] == parts[11]


@pytest.mark.unit
def test_subbasins_of_a_line_of_tributaries_are_equal_and_connected():
    pytest.importorskip("pymetis")
    n = 24
    graph = line_graph(np.full(n, 10))
    parts = partition.assign_subbasins(
        graph,
        np.ones(n, dtype=bool),
        np.arange(n),
        4,
        min_subtree_size=1,
        imbalance_target=1.0,
    )
    assert np.array_equal(np.bincount(parts, minlength=4), np.full(4, 6))
    for part in range(4):
        assert np.all(np.diff(np.flatnonzero(parts == part)) == 1)


# --- rasters ---------------------------------------------------------------------


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


def check_plan(flw, plan):
    parts = plan.parts.ravel()
    mask = flw.mask.ravel()
    fifth = sum(stem.mainstem.size for stem in plan.stems)
    assert plan.loads.sum() + fifth == flw.ncells
    assert np.all(parts[mask] >= 0)
    for stem in plan.stems:
        assert stem.cut_outlets.size == stem.cut_inlets.size == stem.cut_ranks.size
        assert stem.mainstem.size > 0
        assert np.all(np.isin(stem.cut_inlets, stem.mainstem))
        positions = {int(cell): index for index, cell in enumerate(stem.mainstem)}
        assert np.all(np.diff([positions[int(c)] for c in stem.cut_inlets]) >= 0)
        if stem.predecessor >= 0:
            assert flw.idxs_ds[stem.predecessor] == stem.mainstem[0]
            assert parts[stem.predecessor] == stem.trunk
    # flow leaves a part only into a fifth region
    cells = np.flatnonzero((parts >= 0) & (parts < partition.N_TRUNKS))
    downstream = flw.idxs_ds[cells]
    crossing = parts[downstream] != parts[cells]
    assert np.all(parts[downstream[crossing]] == pyflwdir.MAINSTEM)


@pytest.mark.unit
def test_subbasin_plan_has_four_trunks_and_one_accounted_mainstem():
    pytest.importorskip("pymetis")
    flw = tributaries_to_mainstem()
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    parts = plan.parts.ravel()
    assert set(np.unique(parts[flw.mask.ravel()])) == {0, 1, 2, 3, pyflwdir.MAINSTEM}
    assert len(plan.stems) == 1
    check_plan(flw, plan)
    repeated = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    assert np.array_equal(repeated.parts, plan.parts)
    assert np.array_equal(repeated.cut_outlets, plan.cut_outlets)
    assert np.array_equal(repeated.mainstem, plan.mainstem)


@pytest.mark.unit
def test_subbasin_parts_are_balanced_and_connected():
    pytest.importorskip("pymetis")
    flw = tributaries_to_mainstem(nrow=61, ncol=61)
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    assert plan.loads.max() / plan.loads.mean() <= 1.04
    for part in range(4):
        assert ground_pieces(flw, plan.parts.ravel(), part) == 1


@pytest.mark.unit
def test_two_large_basins_are_both_opened_to_balance():
    pytest.importorskip("pymetis")
    # 41 x 41 and 41 x 27: both larger than an equal share of the 41 x 68 raster
    flw = rivers(41, [41, 27])
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    assert len(plan.stems) == 2
    check_plan(flw, plan)
    assert plan.loads.max() / plan.loads.mean() <= 1.04
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    assert np.array_equal(accumulate_by_plan(flw, plan, data), flw.accuflux(data))


@pytest.mark.unit
def test_four_trunks_and_mainstem_walk_match_serial_accumulation():
    pytest.importorskip("pymetis")
    flw = tributaries_to_mainstem()
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    assert np.array_equal(accumulate_by_plan(flw, plan, data), flw.accuflux(data))


@pytest.mark.unit
def test_mainstem_nodata_stops_flow_and_keeps_later_cuts():
    idxs_ds = np.array([1, 2, 2, 1, 2], dtype=np.int32)
    seq = np.array([2, 1, 0, 4, 3], dtype=np.int32)
    data = np.array([5, -9999, 7, 11, 13], dtype=np.int64)
    serial = streams.accuflux(idxs_ds, seq, data, -9999)
    accu = data.copy()

    streams.accuflux_subbasin_mainstem(
        mainstem=np.array([0, 1, 2], dtype=np.int32),
        predecessor=-1,
        cut_outlets=np.array([3, 4], dtype=np.int32),
        cut_inlets=np.array([1, 2], dtype=np.int32),
        data=data,
        accu=accu,
        nodata=-9999,
    )

    assert np.array_equal(accu, serial)
    assert accu[2] == 20


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
    pytest.importorskip("pymetis")
    flw = rivers(31, [31, 21])
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
    plan, _ = pyflwdir.parallel._plan(flw, 4, "subbasin", 1, 1.005)
    assert plan.mainstem.size > 0
    assert set(np.unique(plan.parts)) == {0, 1, 2, 3, pyflwdir.MAINSTEM}


@pytest.mark.unit
def test_mainstem_cells_weigh_with_the_tributary_entering_them():
    stem = np.array([-1, 0, 0, 0, 0])
    position = np.array([-1, 50, 51, 52, 53])
    weights = partition._mainstem_weights(stem, position, np.array([100]))
    assert np.array_equal(weights, [0, 51, 1, 1, 47])
    weights = partition._mainstem_weights(stem, position, np.array([51]))
    assert np.array_equal(weights, [0, 51, 0, 0, 0])
    parts = np.array([0, 0, 1, 1, 2])
    assert np.array_equal(
        partition._p_mins(parts, stem, position, np.array([100])), [51]
    )


def side_tributaries(length=100, first=50, count=4):
    """A mainstem column with one-cell tributaries entering it from the east."""
    d8 = np.full((length, 2), 247, dtype=np.uint8)
    d8[:-1, 0] = 4
    d8[-1, 0] = 0
    d8[first : first + count, 1] = 16
    return pyflwdir.from_array(d8, ftype="d8")


@pytest.mark.unit
def test_trunk_keeps_its_mainstem_and_the_fifth_region_stays():
    pytest.importorskip("pymetis")
    flw = side_tributaries()
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    check_plan(flw, plan)
    assert len(plan.stems) == 1
    assert plan.loads.max() <= 52 and plan.mainstem.size >= 47
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    assert np.array_equal(accumulate_by_plan(flw, plan, data), flw.accuflux(data))


@pytest.mark.unit
def test_basins_are_opened_when_method_1_is_unequal():
    pytest.importorskip("pymetis")
    # five equal basins: none is larger than a share, Method 1 is 231 x [1, 1, 1, 2]
    flw = rivers(21, [11] * 5)
    basin = partition.partition_plan(flw, level="basin")
    assert basin.loads.max() / basin.loads.mean() == pytest.approx(1.6)
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    check_plan(flw, plan)
    assert len(plan.stems) >= 1
    assert plan.loads.max() / plan.loads.mean() <= 1.05
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    assert np.array_equal(accumulate_by_plan(flw, plan, data), flw.accuflux(data))


@pytest.mark.unit
def test_an_archipelago_is_divided_by_proximity():
    pytest.importorskip("pymetis")
    d8 = np.full((1, 39), 247, dtype=np.uint8)
    d8[0, ::2] = 0
    flw = pyflwdir.from_array(d8, ftype="d8")
    plan = partition.partition_plan(flw, level="basin")
    assert np.array_equal(np.sort(plan.loads), [5, 5, 5, 5])
    for part in range(4):
        columns = np.flatnonzero(plan.parts[0] == part)
        assert columns.max() - columns.min() == 8


@pytest.mark.unit
def test_metis_never_leaves_a_part_empty():
    pytest.importorskip("pymetis")
    graph = line_graph([101, 1])
    parts = partition._metis_parts(graph, 2, np.array([0.5, 0.5]), 42, 1.0)
    assert sorted(parts) == [0, 1]
    graph = line_graph([1, 1, 5, 1, 2])
    parts = partition._fill_empty_parts(graph, np.zeros(5, dtype=np.int32), 2)
    assert np.array_equal(parts, [0, 0, 0, 0, 1]) or np.array_equal(parts, [1, 0, 0, 0, 0])


@pytest.mark.unit
def test_nodes_without_cells_follow_a_neighbour():
    pytest.importorskip("pymetis")
    weights = np.array([1] * 20 + [0], dtype=np.int64)
    rows = np.zeros(21)
    cols = np.r_[np.arange(20) * 2.0, np.nan]
    edges = np.array([[19, 20]])
    graph = partition.PartitionGraph(weights, rows, cols, edges, np.ones(1, dtype=np.int64))
    parts = partition.assign_basins(graph, 4)
    assert np.array_equal(np.bincount(parts[:20], minlength=4), [5, 5, 5, 5])
    assert parts[20] == parts[19]


@pytest.mark.unit
def test_two_tributaries_entering_one_cell_keep_a_fifth_region():
    pytest.importorskip("pymetis")
    d8 = np.full((100, 3), 247, dtype=np.uint8)
    d8[:-1, 1] = 4
    d8[-1, 1] = 0
    d8[50, 0] = 1
    d8[50, 2] = 16
    flw = pyflwdir.from_array(d8, ftype="d8")
    plan = partition.partition_plan(flw, level="subbasin", min_subtree_size=1)
    check_plan(flw, plan)
    assert np.array_equal(np.sort(plan.loads), [0, 0, 1, 51])
    assert plan.mainstem.size == 50
    data = np.arange(1, flw.size + 1, dtype=np.int64)
    assert np.array_equal(accumulate_by_plan(flw, plan, data), flw.accuflux(data))


def connected_parts(parts, edges):
    """Is every part connected over ``edges``?"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = parts.size
    for part in np.unique(parts):
        members = parts == part
        keep = members[edges[:, 0]] & members[edges[:, 1]]
        matrix = coo_matrix(
            (np.ones(int(keep.sum())), (edges[keep, 0], edges[keep, 1])), shape=(n, n)
        )
        labels = connected_components(matrix, directed=False)[1]
        if np.unique(labels[members]).size > 1:
            return False
    return True


@pytest.mark.unit
def test_a_node_without_cells_still_joins_its_neighbours():
    pytest.importorskip("pymetis")
    # a hub without cells (and so without a centroid) is all that joins six basins
    weights = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    rows = np.zeros(7)
    cols = np.array([np.nan, 0, 10, 20, 30, 10.1, 20.1])
    edges = np.column_stack((np.zeros(6, dtype=np.int64), np.arange(1, 7)))
    graph = partition.PartitionGraph(weights, rows, cols, edges, np.ones(6, dtype=np.int64))
    parts = partition.assign_basins(graph, 4)
    assert np.unique(parts).size == 4
    assert connected_parts(parts, edges)


@pytest.mark.unit
def test_refinement_keeps_parts_joined_across_the_mainstem():
    pytest.importorskip("pymetis")
    weights = np.array([22, 14, 11, 3, 24, 9, 17, 12], dtype=np.int64)
    ground = np.array([[0, 3], [0, 6], [1, 4], [2, 6], [3, 7], [4, 5], [4, 6]])
    graph = partition.PartitionGraph(
        weights, np.zeros(8), np.arange(8, dtype=np.float64), ground,
        np.ones(ground.shape[0], dtype=np.int64),
    )
    is_tributary = np.arange(8) > 0
    position = np.arange(8)
    parts = partition.assign_subbasins(
        graph, is_tributary, position, 4, stem_lengths=np.array([8]), min_subtree_size=1
    )
    banks = np.column_stack((np.arange(1, 7), np.arange(2, 8)))
    assert connected_parts(parts, np.concatenate((ground, banks)))


@pytest.mark.unit
def test_basins_cut_off_by_a_dominant_basin_without_a_part_join_it():
    pytest.importorskip("pymetis")
    # a dominant hub with four one-cell basins around it, and three parts besides
    weights = np.array([10, 1, 1, 1, 1], dtype=np.int64)
    edges = np.column_stack((np.zeros(4, dtype=np.int64), np.arange(1, 5)))
    graph = partition.PartitionGraph(
        weights, np.zeros(5), np.array([0.0, -1, 1, 2, 3]), edges, np.ones(4, dtype=np.int64)
    )
    parts = partition.assign_basins(graph, 4)
    assert np.unique(parts).size == 4
    assert connected_parts(parts, edges)


@pytest.mark.unit
def test_a_cell_less_tributary_keeps_its_banks_joined():
    pytest.importorskip("pymetis")
    weights = np.array([24, 2, 0, 1] + [2] * 31, dtype=np.int64)
    ground = np.array([[0, 2], [1, 2], [2, 3], [1, 4]] + [[k, k + 1] for k in range(4, 34)])
    cols = np.arange(35, dtype=np.float64)
    cols[2] = np.nan
    graph = partition.PartitionGraph(
        weights, np.zeros(35), cols, ground, np.ones(ground.shape[0], dtype=np.int64)
    )
    is_tributary = np.arange(35) > 0
    position = np.full(35, -1)
    position[4:] = np.arange(31)
    position[[1, 2, 3]] = [31, 32, 33]
    parts = partition.assign_subbasins(graph, is_tributary, position, 4)
    order = np.r_[np.arange(4, 35), 1, 2, 3]
    banks = np.column_stack((order[:-1], order[1:]))
    assert connected_parts(parts, np.concatenate((ground, banks)))


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
