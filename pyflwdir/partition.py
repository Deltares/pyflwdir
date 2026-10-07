"""Spatial FlowTopo partitions for process-level parallelism.

The basin graph, METIS island handling, dominant-basin refinement and 4+1
mainstem execution follow the current FlowTopo C implementation:
https://doi.org/10.5281/zenodo.22227621. Tributary allocation additionally
coarsens small neighbouring tributaries and applies contiguous weighted METIS
so that single-basin domains retain coherent process regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import permutations
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from numba import njit

from . import core, streams

if TYPE_CHECKING:
    from .flwdir import Flwdir

N_TRUNKS = 4
MAINSTEM = 4
"""Logical fifth subregion, processed after the four parallel trunks."""

logger = logging.getLogger(__name__)


@dataclass
class PartitionPlan:
    """Internal execution data for a basin or subbasin partition."""

    parts: np.ndarray
    loads: np.ndarray
    basin_ids: np.ndarray
    level: str
    cut_outlets: np.ndarray
    cut_inlets: np.ndarray
    cut_ranks: np.ndarray
    mainstem: np.ndarray
    predecessor: int
    max_rank: int


def _require_pymetis():
    try:
        import pymetis
    except ImportError as error:
        raise ImportError(
            "METIS partitioning requires the optional 'partition' dependency. "
            "Install it with `pip install pyflwdir[partition]`."
        ) from error
    return pymetis


def _raster_shape(flw: "Flwdir") -> tuple[int, int]:
    shape = flw.shape
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("METIS partitioning requires a FlwdirRaster.")
    return int(shape[0]), int(shape[1])


def _basin_graph(
    basin_ids: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Return basin labels, weights, centroids and 4-neighbour adjacency."""
    basin_ids = basin_ids.reshape(shape)
    mask = mask.reshape(shape)
    labels, weights = np.unique(basin_ids[mask & (basin_ids > 0)], return_counts=True)
    labels = labels.astype(np.int64, copy=False)
    weights = weights.astype(np.int64, copy=False)
    if labels.size == 0:
        return (
            labels,
            weights,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            [],
        )

    lookup = np.full(int(labels[-1]) + 1, -1, dtype=np.int64)
    lookup[labels] = np.arange(labels.size)
    flat = basin_ids.ravel()
    valid = mask.ravel() & (flat > 0)
    nodes = np.full(flat.size, -1, dtype=np.int64)
    nodes[valid] = lookup[flat[valid]]
    rows, cols = np.indices(shape)
    centroid_row = (
        np.bincount(nodes[valid], weights=rows.ravel()[valid], minlength=labels.size)
        / weights
    )
    centroid_col = (
        np.bincount(nodes[valid], weights=cols.ravel()[valid], minlength=labels.size)
        / weights
    )

    node_grid = nodes.reshape(shape)
    pairs = []
    for first, second in (
        (node_grid[:, :-1], node_grid[:, 1:]),
        (node_grid[:-1, :], node_grid[1:, :]),
    ):
        edge = (first >= 0) & (second >= 0) & (first != second)
        if np.any(edge):
            a = first[edge]
            b = second[edge]
            pairs.append(np.column_stack((np.minimum(a, b), np.maximum(a, b))))
    if pairs:
        edges = np.unique(np.concatenate(pairs, axis=0), axis=0)
    else:
        edges = np.empty((0, 2), dtype=np.int64)

    adjacency = [[] for _ in range(labels.size)]
    for first, second in edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    return labels, weights, centroid_row, centroid_col, adjacency


def _components(adjacency: list[list[int]]) -> list[np.ndarray]:
    component = np.full(len(adjacency), -1, dtype=np.int32)
    groups = []
    for start in range(len(adjacency)):
        if component[start] >= 0:
            continue
        number = len(groups)
        queue = [start]
        component[start] = number
        members = []
        while queue:
            node = queue.pop()
            members.append(node)
            for neighbour in adjacency[node]:
                if component[neighbour] < 0:
                    component[neighbour] = number
                    queue.append(neighbour)
        groups.append(np.asarray(members, dtype=np.int64))
    return groups


def _greedy_dominant(weights: np.ndarray, n_parts: int, dominant: int) -> np.ndarray:
    assignment = np.full(weights.size, -1, dtype=np.int32)
    assignment[dominant] = 0
    load = np.zeros(n_parts, dtype=np.int64)
    load[0] = weights[dominant]
    remaining = np.flatnonzero(np.arange(weights.size) != dominant)
    for item in remaining[np.argsort(-weights[remaining], kind="stable")]:
        if n_parts > 1:
            part = 1 + int(np.argmin(load[1:]))
        else:
            part = 0
        assignment[item] = part
        load[part] += weights[item]
    return assignment


def _metis_component(
    adjacency: list[list[int]],
    weights: np.ndarray,
    members: np.ndarray,
    n_parts: int,
    seed: int,
) -> np.ndarray:
    pymetis = _require_pymetis()
    local = {int(old): new for new, old in enumerate(members)}
    subgraph = [
        [local[neighbour] for neighbour in adjacency[int(old)] if neighbour in local]
        for old in members
    ]
    subweights = weights[members]
    target = float(subweights.sum()) / n_parts
    imbalance = max(1.05, float(subweights.max()) / target * 1.05)
    ufactor = max(1, int(round((imbalance - 1.0) * 1000)))
    options = pymetis.Options(seed=seed, ufactor=ufactor, contig=1)
    result = pymetis.part_graph(
        n_parts,
        adjacency=subgraph,
        vweights=subweights.tolist(),
        recursive=False,
        options=options,
    )
    return np.asarray(result.vertex_part, dtype=np.int32)


def _assign_islands(
    assignment: np.ndarray,
    components: list[np.ndarray],
    mainland_index: int,
    weights: np.ndarray,
    centroid_row: np.ndarray,
    centroid_col: np.ndarray,
    n_parts: int,
    cap_factor: float,
) -> None:
    mainland = components[mainland_index]
    load = np.bincount(
        assignment[mainland], weights=weights[mainland], minlength=n_parts
    ).astype(np.int64)
    target = float(weights.sum()) / n_parts
    cap = cap_factor * target
    islands = [
        component
        for index, component in enumerate(components)
        if index != mainland_index
    ]
    islands.sort(key=lambda nodes: int(weights[nodes].sum()), reverse=True)
    for island in islands:
        island_row = float(centroid_row[island].mean())
        island_col = float(centroid_col[island].mean())
        best_part = -1
        best_distance = np.inf
        for part in range(n_parts):
            if load[part] >= cap:
                continue
            candidates = mainland[assignment[mainland] == part]
            if candidates.size == 0:
                continue
            distance = np.min(
                (centroid_row[candidates] - island_row) ** 2
                + (centroid_col[candidates] - island_col) ** 2
            )
            if distance < best_distance:
                best_distance = float(distance)
                best_part = part
        if best_part < 0:
            best_part = int(np.argmin(load))
        assignment[island] = best_part
        load[best_part] += int(weights[island].sum())


def _boundary_refine(
    assignment: np.ndarray,
    adjacency: list[list[int]],
    weights: np.ndarray,
    centroid_row: np.ndarray,
    centroid_col: np.ndarray,
    n_parts: int,
) -> None:
    """Move small adjacent basins toward under-loaded partitions."""
    target = int(weights.sum()) // n_parts
    tolerance = int(0.05 * target)
    max_movable = int(0.25 * target)
    from_floor = int(0.30 * target)
    load = np.bincount(assignment, weights=weights, minlength=n_parts).astype(np.int64)

    for _ in range(weights.size):
        deficits = target - load
        under = int(np.argmax(deficits))
        if deficits[under] <= tolerance:
            break
        rank_weight = np.bincount(
            assignment,
            weights=weights,
            minlength=n_parts,
        ).astype(np.float64)
        row_sum = np.bincount(
            assignment,
            weights=centroid_row * weights,
            minlength=n_parts,
        )
        col_sum = np.bincount(
            assignment,
            weights=centroid_col * weights,
            minlength=n_parts,
        )
        rank_row = np.divide(
            row_sum, rank_weight, out=np.zeros(n_parts), where=rank_weight > 0
        )
        rank_col = np.divide(
            col_sum, rank_weight, out=np.zeros(n_parts), where=rank_weight > 0
        )

        best = -1
        best_distance = np.inf
        for basin in range(weights.size):
            source = int(assignment[basin])
            weight = int(weights[basin])
            if source == under or load[source] <= target:
                continue
            if weight > max_movable:
                continue
            if load[under] + weight > target + tolerance:
                continue
            if load[source] - weight < from_floor:
                continue
            if not any(
                assignment[neighbour] == under for neighbour in adjacency[basin]
            ):
                continue
            distance = (centroid_row[basin] - rank_row[under]) ** 2 + (
                centroid_col[basin] - rank_col[under]
            ) ** 2
            if distance < best_distance:
                best_distance = float(distance)
                best = basin
        if best < 0:
            break
        source = int(assignment[best])
        assignment[best] = under
        load[source] -= weights[best]
        load[under] += weights[best]


def _basin_partition(
    flw: "Flwdir",
    n_parts: int,
    seed: int,
    refine: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = _raster_shape(flw)
    mask = flw.mask.ravel()
    basin_ids = flw.basins().ravel()
    labels, weights, centroid_row, centroid_col, adjacency = _basin_graph(
        basin_ids.reshape(shape), mask.reshape(shape), shape
    )
    parts = np.full(flw.size, -1, dtype=np.int32)
    if labels.size == 0:
        return parts, np.zeros(n_parts, dtype=np.int64), basin_ids

    if n_parts == 1:
        assignment = np.zeros(labels.size, dtype=np.int32)
    elif labels.size < n_parts:
        assignment = np.arange(labels.size, dtype=np.int32)
    else:
        dominant = int(np.argmax(weights))
        if weights[dominant] / weights.sum() > 0.70:
            assignment = _greedy_dominant(weights, n_parts, dominant)
        else:
            components = _components(adjacency)
            mainland_index = int(
                np.argmax([weights[component].sum() for component in components])
            )
            mainland = components[mainland_index]
            assignment = np.full(labels.size, -1, dtype=np.int32)
            try:
                if mainland.size < n_parts:
                    raise ValueError("mainland has fewer basins than partitions")
                assignment[mainland] = _metis_component(
                    adjacency, weights, mainland, n_parts, seed
                )
                _assign_islands(
                    assignment,
                    components,
                    mainland_index,
                    weights,
                    centroid_row,
                    centroid_col,
                    n_parts,
                    cap_factor=1.5,
                )
            except (RuntimeError, ValueError) as error:
                logger.warning(
                    "METIS partitioning failed; using the C implementation's "
                    "round-robin fallback: %s",
                    error,
                )
                assignment = np.arange(labels.size, dtype=np.int32) % n_parts
        if refine:
            _boundary_refine(
                assignment,
                adjacency,
                weights,
                centroid_row,
                centroid_col,
                n_parts,
            )

    lookup = np.full(int(labels[-1]) + 1, -1, dtype=np.int32)
    lookup[labels] = assignment
    valid = mask & (basin_ids > 0)
    parts[valid] = lookup[basin_ids[valid]]
    load = np.bincount(parts[parts >= 0], minlength=n_parts).astype(np.int64)
    return parts, load, basin_ids


@njit(cache=True)
def _label_tributaries(
    idxs_ds: np.ndarray,
    seq_d2u: np.ndarray,
    in_dominant: np.ndarray,
    on_mainstem: np.ndarray,
) -> np.ndarray:
    root = np.full(idxs_ds.size, -1, dtype=np.int64)
    for idx in seq_d2u:
        if not in_dominant[idx] or on_mainstem[idx]:
            continue
        downstream = idxs_ds[idx]
        if on_mainstem[downstream]:
            root[idx] = idx
        else:
            root[idx] = root[downstream]
    return root


def _trace_mainstem(
    flw: "Flwdir",
    pit: int,
    upstream: np.ndarray,
    in_dominant: np.ndarray,
) -> np.ndarray:
    indptr, idxs_us = core.upstream_csr(flw.idxs_ds, flw._mv)
    stem = [pit]
    cell = pit
    while True:
        donors = idxs_us[indptr[cell] : indptr[cell + 1]]
        donors = donors[in_dominant[donors]]
        if donors.size == 0:
            break
        cell = int(donors[np.argmax(upstream[donors])])
        stem.append(cell)
    return np.asarray(stem[::-1], dtype=flw.idxs_ds.dtype)


def _seed_empty_rank_centroids(
    records: list[dict],
    rank_rows: np.ndarray,
    rank_cols: np.ndarray,
    rank_occupied: np.ndarray,
    recipient_ranks: list[int],
    cap: np.ndarray,
    shape: tuple[int, int],
    min_subtree_size: int,
) -> None:
    """Seed empty ranks with spatially separated eligible tributaries."""
    anchors = [
        (rank_rows[rank] / shape[0], rank_cols[rank] / shape[1])
        for rank in range(N_TRUNKS)
        if rank_occupied[rank]
    ]
    used_roots: set[int] = set()
    for rank in recipient_ranks:
        if rank_occupied[rank]:
            continue
        candidates = [
            record
            for record in records
            if record["root"] not in used_roots
            and record["size"] >= min_subtree_size
            and record["size"] <= cap[rank]
        ]
        if not candidates:
            continue

        def separation(record: dict) -> tuple[float, int, int]:
            row = record["row"] / shape[0]
            col = record["col"] / shape[1]
            distance = min(
                np.hypot(row - anchor_row, col - anchor_col)
                for anchor_row, anchor_col in anchors
            )
            return float(distance), record["size"], record["position"]

        seed = max(candidates, key=separation)
        rank_rows[rank] = seed["row"]
        rank_cols[rank] = seed["col"]
        anchors.append((seed["row"] / shape[0], seed["col"] / shape[1]))
        used_roots.add(seed["root"])


def _tributary_adjacency(
    records: list[dict],
    roots: np.ndarray,
    shape: tuple[int, int],
) -> list[set[int]]:
    """Return tributary neighbours across boundaries and along the mainstem."""
    adjacency = [set() for _ in records]
    root_to_record = {record["root"]: index for index, record in enumerate(records)}
    labels = roots.reshape(shape)
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        boundary = (first >= 0) & (second >= 0) & (first != second)
        pairs = np.column_stack((first[boundary], second[boundary]))
        if pairs.size == 0:
            continue
        pairs.sort(axis=1)
        for first_root, second_root in np.unique(pairs, axis=0):
            first_index = root_to_record.get(int(first_root))
            second_index = root_to_record.get(int(second_root))
            if first_index is None or second_index is None:
                continue
            adjacency[first_index].add(second_index)
            adjacency[second_index].add(first_index)

    ordered = sorted(range(len(records)), key=lambda index: records[index]["position"])
    for first_index, second_index in zip(ordered[:-1], ordered[1:]):
        adjacency[first_index].add(second_index)
        adjacency[second_index].add(first_index)
    return adjacency


def _assign_tributary_records(
    records: list[dict],
    adjacency: list[set[int]],
    load: np.ndarray,
    rank_rows: np.ndarray,
    rank_cols: np.ndarray,
    max_rank: int,
    shape: tuple[int, int],
    min_subtree_size: int,
    imbalance_target: float,
) -> np.ndarray:
    """Grow spatially connected tributary regions within the load bound."""
    mean_load = float(load.sum()) / N_TRUNKS
    max_load = int(np.floor(mean_load * imbalance_target))
    cap = np.maximum(max_load - load, 0).astype(np.int64)
    cap[max_rank] = 0
    recipient_ranks = [
        rank for rank in range(N_TRUNKS) if rank != max_rank and cap[rank] > 0
    ]
    current_load = load.copy()
    eligible = {
        index
        for index, record in enumerate(records)
        if record["size"] >= min_subtree_size
    }
    assigned: dict[int, int] = {}
    owned = {rank: set() for rank in recipient_ranks}

    for rank in recipient_ranks:
        candidates = [
            index
            for index in eligible - assigned.keys()
            if records[index]["size"] <= cap[rank]
        ]
        if not candidates:
            continue
        seed = min(
            candidates,
            key=lambda index: (
                np.hypot(
                    (records[index]["row"] - rank_rows[rank]) / shape[0],
                    (records[index]["col"] - rank_cols[rank]) / shape[1],
                ),
                -records[index]["size"],
                -records[index]["position"],
            ),
        )
        record = records[seed]
        assigned[seed] = rank
        owned[rank].add(seed)
        cap[rank] -= record["size"]
        current_load[rank] += record["size"]
        current_load[max_rank] -= record["size"]

    minimum_ratio = max(0.0, 2.0 - imbalance_target)
    while assigned:
        if (
            current_load.max() / mean_load <= imbalance_target
            and current_load.min() / mean_load >= minimum_ratio
        ):
            break
        progressed = False
        for rank in sorted(
            recipient_ranks, key=lambda item: (current_load[item], item)
        ):
            frontier: set[int] = set()
            for index in owned[rank]:
                frontier.update(adjacency[index])
            candidates = [
                index
                for index in frontier - assigned.keys()
                if index in eligible and records[index]["size"] <= cap[rank]
            ]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda index: (
                    np.hypot(
                        (records[index]["row"] - rank_rows[rank]) / shape[0],
                        (records[index]["col"] - rank_cols[rank]) / shape[1],
                    ),
                    -records[index]["size"],
                    -records[index]["position"],
                ),
            )
            record = records[selected]
            assigned[selected] = rank
            owned[rank].add(selected)
            cap[rank] -= record["size"]
            current_load[rank] += record["size"]
            current_load[max_rank] -= record["size"]
            progressed = True
            break
        if not progressed:
            break

    for index, rank in assigned.items():
        records[index]["rank"] = rank
    return current_load


def _partition_tributary_graph(
    records: list[dict],
    adjacency: list[set[int]],
    load: np.ndarray,
    rank_rows: np.ndarray,
    rank_cols: np.ndarray,
    max_rank: int,
    target: int,
    shape: tuple[int, int],
    min_subtree_size: int,
    imbalance_target: float,
) -> bool:
    """Partition eligible tributaries into contiguous, weighted graph regions."""
    eligible = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if record["size"] >= min_subtree_size
        ],
        dtype=np.int64,
    )
    if eligible.size < N_TRUNKS:
        return False

    weights = np.asarray([record["size"] for record in records], dtype=np.int64)
    owner = np.full(len(records), -1, dtype=np.int32)
    queue: list[tuple[int, int, int]] = []
    for supernode, record_index in enumerate(eligible):
        owner[record_index] = supernode
        heappush(queue, (0, supernode, int(record_index)))
    while queue:
        distance, supernode, record_index = heappop(queue)
        if owner[record_index] != supernode:
            continue
        for neighbour in adjacency[record_index]:
            if owner[neighbour] < 0:
                owner[neighbour] = supernode
                heappush(queue, (distance + 1, supernode, neighbour))
    if np.any(owner < 0):
        raise RuntimeError("Tributary adjacency graph is disconnected.")

    super_weights = np.bincount(owner, weights=weights, minlength=eligible.size).astype(
        np.int64
    )
    super_rows = (
        np.bincount(
            owner,
            weights=np.asarray([record["row"] for record in records]) * weights,
            minlength=eligible.size,
        )
        / super_weights
    )
    super_cols = (
        np.bincount(
            owner,
            weights=np.asarray([record["col"] for record in records]) * weights,
            minlength=eligible.size,
        )
        / super_weights
    )
    super_adjacency = [set() for _ in eligible]
    for record_index, neighbours in enumerate(adjacency):
        first = int(owner[record_index])
        for neighbour in neighbours:
            second = int(owner[neighbour])
            if first != second:
                super_adjacency[first].add(second)
                super_adjacency[second].add(first)

    base_load = load.copy()
    base_load[max_rank] -= int(weights.sum())
    desired = np.maximum(float(target) - base_load, 1.0)
    target_weights = desired / desired.sum()

    pymetis = _require_pymetis()
    ufactor = max(1, int(round((imbalance_target - 1.0) * 1000)))
    options = pymetis.Options(seed=0, ufactor=ufactor, contig=1)
    result = pymetis.part_graph(
        N_TRUNKS,
        adjacency=[sorted(neighbours) for neighbours in super_adjacency],
        vweights=super_weights.tolist(),
        tpwgts=target_weights.tolist(),
        recursive=False,
        options=options,
    )
    graph_parts = np.asarray(result.vertex_part, dtype=np.int32)
    graph_load = np.bincount(
        graph_parts, weights=super_weights, minlength=N_TRUNKS
    ).astype(np.int64)
    graph_rows = np.asarray(
        [
            np.average(
                super_rows[graph_parts == part],
                weights=super_weights[graph_parts == part],
            )
            for part in range(N_TRUNKS)
        ]
    )
    graph_cols = np.asarray(
        [
            np.average(
                super_cols[graph_parts == part],
                weights=super_weights[graph_parts == part],
            )
            for part in range(N_TRUNKS)
        ]
    )

    best_mapping = None
    best_score = None
    mean_load = float(load.sum()) / N_TRUNKS
    for mapping in permutations(range(N_TRUNKS)):
        candidate_load = base_load.copy()
        spatial_cost = 0.0
        for graph_part, rank in enumerate(mapping):
            candidate_load[rank] += graph_load[graph_part]
            spatial_cost += np.hypot(
                (graph_rows[graph_part] - rank_rows[rank]) / shape[0],
                (graph_cols[graph_part] - rank_cols[rank]) / shape[1],
            )
        score = (
            float(candidate_load.max() / mean_load),
            int(candidate_load.max() - candidate_load.min()),
            float(spatial_cost),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_mapping = mapping

    assert best_mapping is not None
    for record_index, supernode in enumerate(owner):
        records[record_index]["rank"] = best_mapping[graph_parts[supernode]]
    return True


def _subbasin_partition(
    flw: "Flwdir",
    parts: np.ndarray,
    load: np.ndarray,
    basin_ids: np.ndarray,
    min_subtree_size: int,
    imbalance_target: float,
) -> PartitionPlan:
    target = int(np.count_nonzero(parts >= 0)) // N_TRUNKS
    max_rank = int(np.argmax(load))
    if load[max_rank] <= target:
        return _empty_plan(flw, parts, load, basin_ids, "subbasin", max_rank)

    candidate_ids = basin_ids[parts == max_rank]
    labels, counts = np.unique(candidate_ids[candidate_ids > 0], return_counts=True)
    if labels.size == 0:
        return _empty_plan(flw, parts, load, basin_ids, "subbasin", max_rank)
    dominant_id = int(labels[np.argmax(counts)])
    dominant_size = int(counts.max())
    in_dominant = basin_ids == dominant_id

    seq_d2u = core.idxs_seq_dfs(flw.idxs_ds, flw.idxs_pit, flw._mv)
    upstream = streams.accuflux(
        flw.idxs_ds,
        seq_d2u,
        np.ones(flw.size, dtype=np.int64),
        -1,
    )
    members = np.flatnonzero(in_dominant)
    pits = members[flw.idxs_ds[members] == members]
    pit = int(pits[0]) if pits.size else int(members[np.argmax(upstream[members])])
    mainstem = _trace_mainstem(flw, pit, upstream, in_dominant)
    on_mainstem = np.zeros(flw.size, dtype=np.bool_)
    on_mainstem[mainstem] = True
    mainstem_position = np.full(flw.size, -1, dtype=np.int32)
    mainstem_position[mainstem] = np.arange(mainstem.size, dtype=np.int32)

    roots = _label_tributaries(
        flw.idxs_ds,
        seq_d2u,
        np.ascontiguousarray(in_dominant),
        on_mainstem,
    )
    roots_found = np.unique(roots[roots >= 0])
    if roots_found.size == 0:
        return _empty_plan(flw, parts, load, basin_ids, "subbasin", max_rank)

    shape = _raster_shape(flw)
    row, col = np.indices(shape)
    flat_row = row.ravel()
    flat_col = col.ravel()
    records = []
    for root in roots_found:
        cells = np.flatnonzero(roots == root)
        inlet = int(flw.idxs_ds[root])
        records.append(
            {
                "root": int(root),
                "inlet": inlet,
                "position": int(mainstem_position[inlet]),
                "cells": cells,
                "size": int(cells.size),
                "row": float(flat_row[cells].mean()),
                "col": float(flat_col[cells].mean()),
                "rank": -1,
            }
        )
    records.sort(key=lambda record: (-record["size"], -record["position"]))

    rank_rows = np.full(N_TRUNKS, 0.5 * shape[0], dtype=np.float64)
    rank_cols = np.full(N_TRUNKS, 0.5 * shape[1], dtype=np.float64)
    rank_occupied = np.zeros(N_TRUNKS, dtype=np.bool_)
    for rank in range(N_TRUNKS):
        cells = np.flatnonzero(parts == rank)
        if cells.size:
            rank_rows[rank] = flat_row[cells].mean()
            rank_cols[rank] = flat_col[cells].mean()
            rank_occupied[rank] = True

    mean_load = float(load.sum()) / N_TRUNKS
    max_load = int(np.floor(mean_load * imbalance_target))
    cap = np.maximum(max_load - load, 0).astype(np.int64)
    cap[max_rank] = 0
    recipient_ranks = [
        rank for rank in range(N_TRUNKS) if rank != max_rank and cap[rank] > 0
    ]
    _seed_empty_rank_centroids(
        records,
        rank_rows,
        rank_cols,
        rank_occupied,
        recipient_ranks,
        cap,
        shape,
        min_subtree_size,
    )
    adjacency = _tributary_adjacency(records, roots, shape)
    graph_partitioned = _partition_tributary_graph(
        records,
        adjacency,
        load,
        rank_rows,
        rank_cols,
        max_rank,
        target,
        shape,
        min_subtree_size,
        imbalance_target,
    )
    if not graph_partitioned:
        _assign_tributary_records(
            records,
            adjacency,
            load,
            rank_rows,
            rank_cols,
            max_rank,
            shape,
            min_subtree_size,
            imbalance_target,
        )

    extracted = [
        record
        for record in records
        if record["rank"] >= 0 and record["rank"] != max_rank
    ]
    if not extracted:
        return _empty_plan(flw, parts, load, basin_ids, "subbasin", max_rank)
    p_min = min(record["position"] for record in extracted)

    refined = parts.copy()
    for record in records:
        if record["rank"] >= 0 and record["position"] >= p_min:
            refined[record["cells"]] = record["rank"]
    refined[mainstem[p_min:]] = MAINSTEM
    refined[mainstem[:p_min]] = max_rank

    cuts = sorted(
        (record for record in records if record["position"] >= p_min),
        key=lambda record: record["position"],
    )
    cut_outlets = np.asarray(
        [record["root"] for record in cuts], dtype=flw.idxs_ds.dtype
    )
    cut_inlets = np.asarray(
        [record["inlet"] for record in cuts], dtype=flw.idxs_ds.dtype
    )
    cut_ranks = np.asarray(
        [record["rank"] if record["rank"] >= 0 else max_rank for record in cuts],
        dtype=np.int8,
    )
    predecessor = int(mainstem[p_min - 1]) if p_min > 0 else -1

    accounted = mainstem.size - p_min + sum(record["size"] for record in cuts)
    if predecessor >= 0:
        accounted += int(upstream[predecessor])
    if accounted != dominant_size:
        raise RuntimeError(
            "Subbasin partition does not account for every dominant-basin cell: "
            f"{accounted} != {dominant_size}."
        )

    final_load = np.bincount(
        refined[(refined >= 0) & (refined < N_TRUNKS)], minlength=N_TRUNKS
    ).astype(np.int64)
    return PartitionPlan(
        parts=refined.reshape(flw.shape),
        loads=final_load,
        basin_ids=basin_ids.reshape(flw.shape),
        level="subbasin",
        cut_outlets=cut_outlets,
        cut_inlets=cut_inlets,
        cut_ranks=cut_ranks,
        mainstem=mainstem[p_min:],
        predecessor=predecessor,
        max_rank=max_rank,
    )


def _empty_plan(
    flw: "Flwdir",
    parts: np.ndarray,
    load: np.ndarray,
    basin_ids: np.ndarray,
    level: str,
    max_rank: int = -1,
) -> PartitionPlan:
    empty = np.empty(0, dtype=flw.idxs_ds.dtype)
    return PartitionPlan(
        parts=parts.reshape(flw.shape),
        loads=load,
        basin_ids=basin_ids.reshape(flw.shape),
        level=level,
        cut_outlets=empty,
        cut_inlets=empty.copy(),
        cut_ranks=np.empty(0, dtype=np.int8),
        mainstem=empty.copy(),
        predecessor=-1,
        max_rank=max_rank,
    )


def partition_plan(
    flw: "Flwdir",
    level: Literal["basin", "subbasin"] = "subbasin",
    n_parts: int = N_TRUNKS,
    *,
    seed: int = 42,
    refine: bool = True,
    min_subtree_size: int = 100_000,
    imbalance_target: float = 1.05,
) -> PartitionPlan:
    """Build the full FlowTopo partition and merge plan."""
    if level not in ("basin", "subbasin"):
        raise ValueError("level must be 'basin' or 'subbasin'")
    if n_parts < 1:
        raise ValueError("n_parts must be at least 1")
    if level == "subbasin" and n_parts != N_TRUNKS:
        raise ValueError("FlowTopo subbasin partitioning requires n_parts=4")
    if min_subtree_size < 1:
        raise ValueError("min_subtree_size must be at least 1")
    if imbalance_target < 1:
        raise ValueError("imbalance_target must be at least 1")

    parts, load, basin_ids = _basin_partition(flw, n_parts, seed, refine)
    if level == "basin":
        return _empty_plan(flw, parts, load, basin_ids, level)
    return _subbasin_partition(
        flw,
        parts,
        load,
        basin_ids,
        min_subtree_size,
        imbalance_target,
    )


def partition(
    flw: "Flwdir",
    level: Literal["basin", "subbasin"] = "subbasin",
    n_parts: int = N_TRUNKS,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign raster cells to FlowTopo process-level subregions."""
    plan = partition_plan(flw, level=level, n_parts=n_parts, **kwargs)
    return plan.parts, plan.loads
