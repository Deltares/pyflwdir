"""Spatial FlowTopo partitions for process-level parallelism.

FlowTopo (https://doi.org/10.5281/zenodo.22227621) divides a region into four
process-level parts, at two levels:

* ``"basin"`` (Method 1): every basin stays whole.  The basins are the nodes of
  a graph whose edges are their shared raster boundaries, weighted by the length
  of the boundary.  A basin larger than an equal share is a part of its own,
  with the small basins it cuts off from the others; the other basins share the
  remaining parts.
* ``"subbasin"`` (Method 2): when Method 1's parts are not within
  ``imbalance_target`` of an equal share, basins are opened along their
  mainstems into the tributary subtrees that drain into the mainstem -- the
  largest basin first, then the next, while the parts are still unequal.  Those
  tributaries and the other basins form one graph, divided at once into equal
  parts; each mainstem cell weighs with the tributary that enters it.
  Below ``P_min``, the most upstream mainstem cell where a tributary of another
  part enters, each opened mainstem is a logical fifth region, walked once
  after the four parts; above it, the mainstem stays with the part of its most
  upstream tributaries (its trunk).

Contiguity comes first, balance second: every part is one piece of land.
Land is a large component of the graph -- what touches on the ground, the two
banks of an opened mainstem joined -- and each small component (an island) goes
with its nearest land, a land mass.  The masses get the parts so that the
heaviest part is as light as can be, and no part spans two masses.  A mass is
cut by contiguous weighted METIS, run from several seeds on a graph whose small
tributaries are grouped with a larger neighbour; the most balanced result is
kept, and boundary nodes then move, one tributary or basin at a time, from
heavier to lighter neighbouring parts until every part is within
``imbalance_target`` of its target load.  An island moves whole, to a part
near it.  A part is never cut in two; the odd piece left apart from its part
joins the part around it.

The graph functions take arrays, not a raster, so that a graph built from tiles
(FlowTopo's 90 m region tiles, for one) is partitioned by the same code.
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

METIS_SEEDS = 4
"""METIS runs from this many seeds; see ``_metis_parts`` for the one kept."""

METIS_SLACK = 1.03
"""A METIS result whose parts are within this of their targets is balanced
enough for the refinement; among those, the shortest boundary wins."""

MAX_OPENED_BASINS = 4
"""Method 2 opens at most this many basins along their mainstems."""

ARCHIPELAGO_NEIGHBOURS = 4
"""Without land, each component is linked to this many nearest for METIS; in
the balance refinement, a node of an island may move to the part of this many
nearest nodes of other components."""

ISLAND_PROBES = 16
"""The nearest nodes an island node looks through for those of other components."""

METIS_GROUPS_PER_PART = 8
"""Method 2 groups small tributaries for METIS, but keeps at least this many
groups per part, whatever ``min_subtree_size``."""

P_MIN_ROUNDS = 4
"""Method 2 balances again, with the trunk's mainstem above the new ``P_min``,
at most this many times."""

FRAGMENT_SHARE = 0.01
"""A piece of a part cut off from its main piece and lighter than this share of
an equal part joins the neighbouring part it touches the most."""

logger = logging.getLogger(__name__)


@dataclass
class FifthRegion:
    """One opened basin's mainstem below ``P_min``, walked after the four parts.

    ``cut_inlets`` are in mainstem order; ``predecessor`` is the mainstem cell
    just above ``P_min`` (-1 when ``P_min`` is the source), held by ``trunk``.
    """

    mainstem: np.ndarray
    predecessor: int
    cut_outlets: np.ndarray
    cut_inlets: np.ndarray
    cut_ranks: np.ndarray
    trunk: int


@dataclass
class PartitionPlan:
    """Internal execution data for a basin or subbasin partition."""

    parts: np.ndarray
    loads: np.ndarray
    basin_ids: np.ndarray
    level: str
    stems: tuple[FifthRegion, ...] = ()

    @property
    def mainstem(self) -> np.ndarray:
        """The cells of every fifth region, one opened basin after the other."""
        if not self.stems:
            return np.empty(0, dtype=np.intp)
        return np.concatenate([stem.mainstem for stem in self.stems])

    @property
    def cut_outlets(self) -> np.ndarray:
        if not self.stems:
            return np.empty(0, dtype=np.intp)
        return np.concatenate([stem.cut_outlets for stem in self.stems])

    @property
    def cut_inlets(self) -> np.ndarray:
        if not self.stems:
            return np.empty(0, dtype=np.intp)
        return np.concatenate([stem.cut_inlets for stem in self.stems])

    @property
    def cut_ranks(self) -> np.ndarray:
        if not self.stems:
            return np.empty(0, dtype=np.int8)
        return np.concatenate([stem.cut_ranks for stem in self.stems])


@dataclass
class PartitionGraph:
    """Nodes (whole basins or tributary subtrees) and their shared boundaries.

    ``edges`` holds every adjacent pair once, first node smaller; ``edge_weights``
    is the length of the shared boundary in cell sides.  Centroids are in cell
    rows and columns.
    """

    weights: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    edges: np.ndarray
    edge_weights: np.ndarray

    @property
    def size(self) -> int:
        return int(self.weights.size)


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


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def merge_edges(
    pairs: list[np.ndarray],
    n_nodes: int,
    weights: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique ``(first, second)`` node pairs, first smaller, and their summed weights.

    Each array of ``pairs`` has two columns; a pair without a weight counts 1.
    """
    if not pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    stacked = np.concatenate([np.asarray(pair, dtype=np.int64) for pair in pairs])
    if weights is None:
        counted = np.ones(stacked.shape[0], dtype=np.int64)
    else:
        counted = np.concatenate([np.asarray(w, dtype=np.int64) for w in weights])
    first = np.minimum(stacked[:, 0], stacked[:, 1])
    second = np.maximum(stacked[:, 0], stacked[:, 1])
    keep = first != second
    keys = first[keep] * np.int64(n_nodes) + second[keep]
    unique, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(inverse, weights=counted[keep]).astype(np.int64)
    edges = np.column_stack((unique // n_nodes, unique % n_nodes)).astype(np.int64)
    return edges, summed


def graph_from_node_raster(nodes: np.ndarray, n_nodes: int) -> PartitionGraph:
    """The graph of a 2-D raster of node ids (-1 for cells outside every node).

    Two nodes are adjacent where a cell of one shares a side with a cell of the
    other; the edge weight counts those sides.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    shape = nodes.shape
    flat = nodes.ravel()
    valid = np.flatnonzero(flat >= 0)
    owner = flat[valid]
    weights = np.bincount(owner, minlength=n_nodes).astype(np.int64)
    rows, cols = np.divmod(valid, shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid_row = np.bincount(owner, weights=rows, minlength=n_nodes) / weights
        centroid_col = np.bincount(owner, weights=cols, minlength=n_nodes) / weights
    pairs = []
    for first, second in (
        (nodes[:, :-1], nodes[:, 1:]),
        (nodes[:-1, :], nodes[1:, :]),
    ):
        boundary = (first >= 0) & (second >= 0) & (first != second)
        if np.any(boundary):
            pairs.append(np.column_stack((first[boundary], second[boundary])))
    edges, edge_weights = merge_edges(pairs, n_nodes)
    return PartitionGraph(weights, centroid_row, centroid_col, edges, edge_weights)


def _contract(graph: PartitionGraph, group: np.ndarray, n_groups: int) -> PartitionGraph:
    """The graph whose nodes are the groups of ``graph``'s nodes."""
    weights = np.bincount(group, weights=graph.weights, minlength=n_groups).astype(
        np.int64
    )
    rows = np.bincount(group, weights=graph.rows * graph.weights, minlength=n_groups)
    cols = np.bincount(group, weights=graph.cols * graph.weights, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        rows = rows / weights
        cols = cols / weights
    edges, edge_weights = merge_edges(
        [group[graph.edges]], n_groups, [graph.edge_weights]
    )
    return PartitionGraph(weights, rows, cols, edges, edge_weights)


def _csr(n_nodes: int, edges: np.ndarray, edge_weights: np.ndarray):
    """Symmetric compressed sparse rows of an edge list."""
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    weight = np.concatenate((edge_weights, edge_weights))
    order = np.lexsort((target, source))
    xadj = np.zeros(n_nodes + 1, dtype=np.int64)
    xadj[1:] = np.cumsum(np.bincount(source, minlength=n_nodes))
    return xadj, target[order].astype(np.int64), weight[order].astype(np.int64)


# ---------------------------------------------------------------------------
# METIS and the balance refinement
# ---------------------------------------------------------------------------


def _scaled(values: np.ndarray, limit: int = 2**30) -> np.ndarray:
    """Positive integer weights whose sum stays below METIS's 32-bit range."""
    values = np.asarray(values, dtype=np.int64)
    factor = max(1, int(np.ceil(float(values.sum()) / limit)))
    if factor == 1:
        return np.maximum(values, 1)
    return np.maximum((values + factor - 1) // factor, 1)


def _metis(
    graph: PartitionGraph,
    n_parts: int,
    targets: np.ndarray,
    seed: int,
    ufactor: int,
    contig: bool = True,
) -> np.ndarray:
    pymetis = _require_pymetis()
    xadj, adjncy, eweights = _csr(graph.size, graph.edges, graph.edge_weights)
    options = pymetis.Options(seed=seed, ufactor=ufactor, contig=int(contig))
    result = pymetis.part_graph(
        n_parts,
        adjacency=pymetis.CSRAdjacency(xadj.tolist(), adjncy.tolist()),
        vweights=_scaled(graph.weights).tolist(),
        eweights=_scaled(eweights).tolist(),
        tpwgts=[float(t) for t in targets],
        recursive=False,
        options=options,
    )
    return np.asarray(result.vertex_part, dtype=np.int32), int(result.edge_cuts)


@njit(cache=True)
def _detach(xadj, adjncy, weights, parts, node, part, mark, queue, stamp, limit, out):
    """The nodes that leave ``part`` with ``node``: itself and the peninsulas it holds.

    Without ``node``, its same-part neighbours fall into components.  The body of
    the part -- the one component too large to explore within ``limit`` visited
    nodes, or else the heaviest component -- stays; the other components are
    peninsulas and leave with ``node``.  Returns the number of nodes put into
    ``out`` (``node`` first) and the last stamp used, or -1 when two components
    are too large to explore (``node`` holds the body together).

    ``mark`` holds stamps: ``stamp`` for ``node``, ``stamp + k`` for the k-th
    component of this call.  A search that meets an earlier component of this
    call is part of it; only a component left unexplored can be met that way.
    """
    base = stamp
    mark[node] = base
    out[0] = node
    count = 1
    degree = xadj[node + 1] - xadj[node]
    begins = np.empty(degree + 1, dtype=np.int64)
    ends = np.empty(degree + 1, dtype=np.int64)
    component_weight = np.zeros(degree + 1, dtype=np.float64)
    n_components = 0
    n_large = 0
    for j in range(xadj[node], xadj[node + 1]):
        start = adjncy[j]
        if parts[start] != part or mark[start] > base:
            continue
        n_components += 1
        own = base + n_components
        begin = count
        head = 0
        tail = 0
        queue[tail] = start
        tail += 1
        mark[start] = own
        visited = 0
        merged = False
        too_large = False
        while head < tail:
            current = queue[head]
            head += 1
            visited += 1
            if visited > limit or count >= out.size:
                too_large = True
                break
            out[count] = current
            count += 1
            component_weight[n_components] += weights[current]
            for k in range(xadj[current], xadj[current + 1]):
                neighbour = adjncy[k]
                if neighbour == node or parts[neighbour] != part:
                    continue
                seen = mark[neighbour]
                if seen == own:
                    continue
                if seen > base and seen < own:
                    merged = True
                    break
                mark[neighbour] = own
                queue[tail] = neighbour
                tail += 1
            if merged:
                break
        if merged or too_large:
            # (part of) the body: its nodes are not taken
            count = begin
            begins[n_components] = begin
            ends[n_components] = begin
            component_weight[n_components] = -1.0
            if too_large:
                n_large += 1
                if n_large > 1:
                    return -1, base + n_components
            continue
        begins[n_components] = begin
        ends[n_components] = count
    if n_large == 0 and n_components > 0:
        # every component was explored: the heaviest is the body and stays
        body = 1
        for k in range(2, n_components + 1):
            if component_weight[k] > component_weight[body]:
                body = k
        removed = ends[body] - begins[body]
        for k in range(ends[body], count):
            out[k - removed] = out[k]
        count -= removed
    return count, base + n_components


@njit(cache=True)
def _refine_balance(
    xadj,
    adjncy,
    link_xadj,
    link_adjncy,
    weights,
    parts,
    target_load,
    goal,
    max_moves,
    search_limit,
):
    """Move boundary nodes to lighter neighbouring parts until balanced.

    The objective is the sum over the parts of (load / target - 1) squared: each
    move takes the node, of any part, whose move to a lighter neighbouring part
    (``xadj``, ``adjncy``) lowers that sum the most.  A node whose departure
    would cut a peninsula off its part (``link_xadj``, ``link_adjncy``) takes
    the peninsula with it (``_detach``), so that both parts stay connected; a
    move that would cut the part's body in two is not made.  Load
    can so pass on through a middle part to a light one that does not touch the
    heavy one.  It stops when the largest load-to-target ratio is at most
    ``goal`` or no move lowers the sum.  Returns the number of moves.
    """
    n = weights.size
    n_parts = target_load.size
    load = np.zeros(n_parts, dtype=np.float64)
    for node in range(n):
        load[parts[node]] += weights[node]
    mark = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    stamp = np.int64(1)
    moves = 0
    candidate_node = np.empty(n, dtype=np.int64)
    candidate_part = np.empty(n, dtype=np.int64)
    candidate_gain = np.empty(n, dtype=np.float64)
    while moves < max_moves:
        ratio = load / target_load
        if np.max(ratio) <= goal:
            break
        count = 0
        for node in range(n):
            source = parts[node]
            weight = weights[node]
            if weight >= load[source]:
                continue
            before_source = (ratio[source] - 1.0) ** 2
            after_source = ((load[source] - weight) / target_load[source] - 1.0) ** 2
            best_other = -1
            best_gain = -1e-15
            for j in range(xadj[node], xadj[node + 1]):
                other = parts[adjncy[j]]
                if other == source or ratio[other] >= ratio[source]:
                    continue
                after_other = ((load[other] + weight) / target_load[other] - 1.0) ** 2
                gain = (after_source + after_other) - (
                    before_source + (ratio[other] - 1.0) ** 2
                )
                if gain < best_gain or (
                    best_other >= 0 and gain == best_gain and other < best_other
                ):
                    best_gain = gain
                    best_other = other
            if best_other >= 0:
                candidate_node[count] = node
                candidate_part[count] = best_other
                candidate_gain[count] = best_gain
                count += 1
        if count == 0:
            break
        order = np.argsort(candidate_gain[:count], kind="mergesort")
        moved = False
        for index in order:
            node = candidate_node[index]
            source = parts[node]
            other = candidate_part[index]
            stamp += 1
            taken, stamp = _detach(
                link_xadj,
                link_adjncy,
                weights,
                parts,
                node,
                source,
                mark,
                queue,
                stamp,
                search_limit,
                out,
            )
            if taken < 0:
                continue
            weight = 0.0
            for k in range(taken):
                weight += weights[out[k]]
            if weight >= load[source]:
                continue
            gain = (
                ((load[source] - weight) / target_load[source] - 1.0) ** 2
                + ((load[other] + weight) / target_load[other] - 1.0) ** 2
                - (load[source] / target_load[source] - 1.0) ** 2
                - (load[other] / target_load[other] - 1.0) ** 2
            )
            if gain >= -1e-15:
                continue
            for k in range(taken):
                parts[out[k]] = other
            load[source] -= weight
            load[other] += weight
            moves += 1
            moved = True
            break
        if not moved:
            break
    return moves


def _metis_parts(
    graph: PartitionGraph,
    n_parts: int,
    targets: np.ndarray,
    seed: int,
    imbalance_target: float,
    contig: bool = True,
) -> np.ndarray:
    """Contiguous METIS from several seeds.

    Of the results whose parts are within ``METIS_SLACK`` of their targets --
    near enough for the refinement -- the one with the shortest boundary between
    the parts is kept; when there is none, the most balanced.  With ``contig``,
    ``graph`` must be connected (one land component, see
    ``_partition_components``).
    """
    if graph.size <= n_parts:
        return np.arange(graph.size, dtype=np.int32)
    total = float(graph.weights.sum())
    target_load = np.asarray(targets, dtype=np.float64) * total
    ufactor = max(1, int(round((imbalance_target - 1.0) * 1000)))
    best = None
    best_score = None
    for attempt in range(METIS_SEEDS):
        parts, cut = _metis(graph, n_parts, targets, seed + attempt, ufactor, contig)
        load = np.bincount(parts, weights=graph.weights, minlength=n_parts)
        n_empty = int(np.count_nonzero(np.bincount(parts, minlength=n_parts) == 0))
        ratio = float(np.max(load / target_load))
        score = (n_empty, 0, cut, ratio) if ratio <= METIS_SLACK else (n_empty, 1, ratio, cut)
        if best_score is None or score < best_score:
            best, best_score = parts, score
    assert best is not None
    return _fill_empty_parts(graph, best, n_parts)


def _fill_empty_parts(graph: PartitionGraph, parts: np.ndarray, n_parts: int) -> np.ndarray:
    """An empty part (METIS leaves one with very unequal weights) takes a node of
    the heaviest part that has two or more: the lightest leaf of a spanning tree
    of that part, so that the part stays connected.  ``graph`` has more nodes
    than ``n_parts``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order

    for empty in range(n_parts):
        counts = np.bincount(parts, minlength=n_parts)
        if counts[empty] > 0:
            continue
        load = np.bincount(parts, weights=graph.weights, minlength=n_parts)
        donor = int(np.argmax(np.where(counts > 1, load, -np.inf)))
        members = np.flatnonzero(parts == donor)
        sub = _subgraph(graph, members)
        matrix = coo_matrix(
            (np.ones(sub.edges.shape[0]), (sub.edges[:, 0], sub.edges[:, 1])),
            shape=(sub.size, sub.size),
        ).tocsr()
        _, predecessors = breadth_first_order(matrix, 0, directed=False)
        is_parent = np.zeros(sub.size, dtype=np.bool_)
        is_parent[predecessors[predecessors >= 0]] = True
        leaves = np.flatnonzero(~is_parent)
        leaf = leaves[np.argmin(sub.weights[leaves])]
        parts[members[leaf]] = empty
    return parts


def _island_units(linked: PartitionGraph, parts: np.ndarray, n_parts: int):
    """The units the refinement moves: every node, but an island -- a component
    of ``linked`` lighter than half an equal share -- that is all in one part is
    one unit.  Returns None when there is no such island; else the unit of every
    node, the number of units and the pairs (island unit, unit of one of the
    ``ARCHIPELAGO_NEIGHBOURS`` nearest nodes of other components, among the
    ``ISLAND_PROBES`` nearest of any of the island's nodes)."""
    from scipy.spatial import cKDTree

    component = _components(linked)
    component_weight = np.bincount(component, weights=linked.weights)
    small = component_weight < 0.5 * component_weight.sum() / n_parts
    if component_weight.size < 2 or not small.any():
        return None
    low = np.full(component_weight.size, n_parts, dtype=np.int64)
    high = np.full(component_weight.size, -1, dtype=np.int64)
    np.minimum.at(low, component, parts)
    np.maximum.at(high, component, parts)
    whole = small & (low == high)
    if not whole.any():
        return None
    on_island = whole[component]
    unit = np.empty(linked.size, dtype=np.int64)
    islands = np.flatnonzero(whole)
    loose = np.flatnonzero(~on_island)
    unit[loose] = np.arange(loose.size)
    island_unit = np.full(component_weight.size, -1, dtype=np.int64)
    island_unit[islands] = loose.size + np.arange(islands.size)
    unit[on_island] = island_unit[component[on_island]]
    nodes = np.flatnonzero(on_island)
    points = np.column_stack((linked.rows, linked.cols))
    k = min(ISLAND_PROBES + 1, linked.size)
    _, near = cKDTree(points).query(points[nodes], k=k)
    near = near.reshape(nodes.size, k)
    other = component[near] != component[nodes][:, None]
    keep = other & (np.cumsum(other, axis=1) <= ARCHIPELAGO_NEIGHBOURS)
    reach = np.column_stack(
        (unit[np.repeat(nodes, k)[keep.ravel()]], unit[near.ravel()[keep.ravel()]])
    )
    return unit, loose.size + islands.size, reach


def _adjacency(n_nodes: int, source: np.ndarray, target: np.ndarray):
    """Compressed sparse rows of directed pairs, unique."""
    keys = np.unique(source.astype(np.int64) * n_nodes + target.astype(np.int64))
    source, target = keys // n_nodes, keys % n_nodes
    xadj = np.zeros(n_nodes + 1, dtype=np.int64)
    xadj[1:] = np.cumsum(np.bincount(source, minlength=n_nodes))
    return xadj, target.astype(np.int64)


def _refine(
    graph: PartitionGraph,
    parts: np.ndarray,
    n_parts: int,
    targets: np.ndarray,
    imbalance_target: float,
    linked: PartitionGraph | None = None,
) -> int:
    """Balance ``parts`` in place by moving boundary nodes of ``graph``.

    ``graph`` holds only shared raster boundaries: a node moves only to a part
    it touches on the ground, or a part could creep along an opened mainstem
    from bank to bank.  An island all in one part moves whole, and to the part
    of a node of another component near it (``_island_units``), so that
    islands, and archipelagos, balance the parts without a piece of land apart
    from its part.  Whether a part stays connected is judged on ``linked``
    (``graph`` when None), where the two banks of an opened mainstem touch, as
    they do for METIS.
    """
    if n_parts < 2:
        return 0
    if linked is None:
        linked = graph
    target_load = np.asarray(targets, dtype=np.float64) * float(graph.weights.sum())
    units = _island_units(linked, parts, n_parts)
    unit_parts = parts
    reach = np.empty((0, 2), dtype=np.int64)
    if units is not None:
        unit, n_units, reach = units
        graph = _contract(graph, unit, n_units)
        linked = _contract(linked, unit, n_units)
        unit_parts = np.empty(n_units, dtype=parts.dtype)
        unit_parts[unit] = parts
    xadj, adjncy = _adjacency(
        graph.size,
        np.concatenate((graph.edges[:, 0], graph.edges[:, 1], reach[:, 0])),
        np.concatenate((graph.edges[:, 1], graph.edges[:, 0], reach[:, 1])),
    )
    link_xadj, link_adjncy, _ = _csr(linked.size, linked.edges, linked.edge_weights)
    moves = int(
        _refine_balance(
            xadj,
            adjncy,
            link_xadj,
            link_adjncy,
            graph.weights.astype(np.int64),
            unit_parts,
            target_load,
            float(imbalance_target),
            10 * graph.size,
            50_000,
        )
    )
    if units is not None:
        parts[:] = unit_parts[unit]
    return moves


def _subgraph(graph: PartitionGraph, nodes: np.ndarray) -> PartitionGraph:
    """The graph of ``nodes`` (sorted), renumbered 0 .. nodes.size - 1."""
    index = np.full(graph.size, -1, dtype=np.int64)
    index[nodes] = np.arange(nodes.size)
    keep = (index[graph.edges[:, 0]] >= 0) & (index[graph.edges[:, 1]] >= 0)
    return PartitionGraph(
        graph.weights[nodes],
        graph.rows[nodes],
        graph.cols[nodes],
        index[graph.edges[keep]],
        graph.edge_weights[keep],
    )


def _components(graph: PartitionGraph, keep: np.ndarray | None = None) -> np.ndarray:
    """Connected component of every node, over the edges ``keep`` selects (all)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = graph.size
    edges = graph.edges if keep is None else graph.edges[keep]
    matrix = coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    return connected_components(matrix, directed=False)[1]


def _absorb_fragments(graph: PartitionGraph, parts: np.ndarray, limit: float) -> int:
    """Pieces of a part apart from its heaviest piece and lighter than ``limit``
    join the neighbouring part they share the longest boundary with (in place).

    Tiny tributaries along an opened mainstem can end up so: in their part only
    through the mainstem, after the parts around them moved.  A piece that is
    a component of ``graph`` of its own (an island) stays.  Returns the number
    of nodes moved.
    """
    if graph.size == 0:
        return 0
    first, second = graph.edges[:, 0], graph.edges[:, 1]
    n_parts = int(parts.max()) + 1
    moved_nodes = 0
    for _ in range(8):
        same = parts[first] == parts[second]
        piece = _components(graph, same)
        weight = np.bincount(piece, weights=graph.weights)
        part_of_piece = np.empty(weight.size, dtype=np.int64)
        part_of_piece[piece] = parts
        order = np.lexsort((-weight, part_of_piece))
        _, heads = np.unique(part_of_piece[order], return_index=True)
        candidate = weight < limit
        candidate[order[heads]] = False
        cross = ~same
        pieces = np.concatenate((piece[first[cross]], piece[second[cross]]))
        beyond = np.concatenate((piece[second[cross]], piece[first[cross]]))
        others = np.concatenate((parts[second[cross]], parts[first[cross]]))
        lengths = np.concatenate((graph.edge_weights[cross], graph.edge_weights[cross]))
        # a fragment joins a piece that stays, so that two fragments never swap;
        # one that touches only fragments waits for the next round
        keep = candidate[pieces] & ~candidate[beyond]
        if not keep.any():
            break
        shared = np.zeros((weight.size, n_parts), dtype=np.int64)
        np.add.at(shared, (pieces[keep], others[keep]), lengths[keep])
        absorbed = np.flatnonzero(shared.sum(axis=1) > 0)
        target = np.full(weight.size, -1, dtype=np.int64)
        target[absorbed] = np.argmax(shared[absorbed], axis=1)
        moving = target[piece] >= 0
        parts[moving] = target[piece[moving]]
        moved_nodes += int(np.count_nonzero(moving))
    return moved_nodes


def _live_nodes(graph: PartitionGraph):
    """None when every node has cells and a centroid; else the nodes that do
    (live), and for every node the index of the live node it is merged into (-1
    when its component has none).  A node without cells joins a live node it
    reaches, so that every connection it makes stays."""
    empty = (graph.weights <= 0) | ~np.isfinite(graph.rows) | ~np.isfinite(graph.cols)
    if not np.any(empty):
        return None
    owner = np.where(empty, -1, np.arange(graph.size)).astype(np.int64)
    _follow_neighbours(graph, owner, fill=-1)
    live = np.flatnonzero(~empty)
    index = np.full(graph.size, -1, dtype=np.int64)
    merged = owner >= 0
    index[merged] = np.searchsorted(live, owner[merged])
    return live, index


def _expand(index: np.ndarray, merged_parts: np.ndarray) -> np.ndarray:
    """The parts of the merged graph back on every node; part 0 where none."""
    parts = np.zeros(index.size, dtype=np.int32)
    parts[index >= 0] = merged_parts[index[index >= 0]]
    return parts


def _merge_into(graph: PartitionGraph, nodes: np.ndarray, live: np.ndarray) -> PartitionGraph:
    """``graph`` with every node merged into node ``nodes`` (-1: left out) of a
    graph whose nodes are ``live``, which keep their centroids."""
    keep = nodes >= 0
    weights = np.bincount(
        nodes[keep], weights=graph.weights[keep], minlength=live.size
    ).astype(np.int64)
    pairs = nodes[graph.edges]
    inside = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
    edges, edge_weights = merge_edges(
        [pairs[inside]], live.size, [graph.edge_weights[inside]]
    )
    return PartitionGraph(weights, graph.rows[live], graph.cols[live], edges, edge_weights)


def _follow_neighbours(graph: PartitionGraph, parts: np.ndarray, fill: int = 0) -> None:
    """Nodes without a part (-1) take the part of a neighbour that has one (the
    lowest part when several), step by step; those that reach none take
    ``fill``."""
    first, second = graph.edges[:, 0], graph.edges[:, 1]
    while True:
        missing = parts < 0
        if not np.any(missing):
            return
        forward = missing[first] & ~missing[second]
        backward = missing[second] & ~missing[first]
        if not (np.any(forward) or np.any(backward)):
            parts[missing] = fill
            return
        nodes = np.concatenate((first[forward], second[backward]))
        given = np.concatenate((parts[second[forward]], parts[first[backward]]))
        order = np.lexsort((given, nodes))
        unique, index = np.unique(nodes[order], return_index=True)
        parts[unique] = given[order][index]


def _archipelago(
    graph: PartitionGraph,
    component: np.ndarray,
    component_weight: np.ndarray,
    n_parts: int,
    seed: int,
    imbalance_target: float,
) -> np.ndarray:
    """Parts of a graph none of whose components is half a part: the components,
    each linked to its nearest by centroid, divided by METIS."""
    from scipy.spatial import cKDTree

    n_components = component_weight.size
    if n_components <= n_parts:
        return component.astype(np.int32)
    rows = np.bincount(component, weights=graph.weights * graph.rows) / component_weight
    cols = np.bincount(component, weights=graph.weights * graph.cols) / component_weight
    k = min(ARCHIPELAGO_NEIGHBOURS + 1, n_components)
    centroids = np.column_stack((rows, cols))
    _, nearest = cKDTree(centroids).query(centroids, k=k)
    pairs = np.column_stack(
        (np.repeat(np.arange(n_components), k - 1), nearest[:, 1:].ravel())
    )
    edges, edge_weights = merge_edges([pairs], n_components)
    linked = PartitionGraph(
        component_weight.astype(np.int64), rows, cols, edges, edge_weights
    )
    targets = np.full(n_parts, 1.0 / n_parts)
    parts = _metis_parts(linked, n_parts, targets, seed, imbalance_target, contig=False)
    return parts[component].astype(np.int32)


def _masses(linked: PartitionGraph, component: np.ndarray, land: np.ndarray):
    """Each island component tied to its nearest land node: the virtual edges,
    one per island (from its member nearest to land), and the land component
    every component belongs to (its mass)."""
    from scipy.spatial import cKDTree

    is_land = np.zeros(component.max() + 1, dtype=np.bool_)
    is_land[land] = True
    on_land = np.flatnonzero(is_land[component])
    off_land = np.flatnonzero(~is_land[component])
    mass_of = np.arange(is_land.size)
    if off_land.size == 0:
        return np.empty((0, 2), dtype=np.int64), mass_of
    tree = cKDTree(np.column_stack((linked.rows[on_land], linked.cols[on_land])))
    distance, nearest = tree.query(
        np.column_stack((linked.rows[off_land], linked.cols[off_land]))
    )
    island = component[off_land]
    order = np.lexsort((off_land, distance, island))
    _, first = np.unique(island[order], return_index=True)
    members = off_land[order[first]]
    anchors = on_land[nearest[order[first]]]
    mass_of[component[members]] = component[anchors]
    return np.column_stack((members, anchors)), mass_of


def _carve(
    graph: PartitionGraph,
    nodes: np.ndarray,
    group: np.ndarray,
    targets: np.ndarray,
    seed: int,
    imbalance_target: float,
) -> np.ndarray:
    """Contiguous METIS of ``nodes`` (connected in ``graph``), grouped by
    ``group``, into ``targets.size`` pieces of those shares."""
    if targets.size == 1:
        return np.zeros(nodes.size, dtype=np.int32)
    sub = _subgraph(graph, nodes)
    local_groups, local_group = np.unique(group[nodes], return_inverse=True)
    coarse = _contract(sub, local_group, local_groups.size)
    return _metis_parts(coarse, targets.size, targets, seed, imbalance_target)[local_group]


def _choose_land(
    component_weight: np.ndarray, capacity: np.ndarray, n_parts: int
) -> np.ndarray:
    """The components that get parts of their own (land), the largest first.

    Those of at least half an equal share, at most ``n_parts``; with too few
    nodes on them for every part (``capacity``), the next largest join.  None
    when no component is that large (an archipelago).
    """
    share = component_weight.sum() / n_parts
    order = np.argsort(-component_weight, kind="stable")
    n_land = min(int(np.count_nonzero(component_weight >= 0.5 * share)), n_parts)
    if n_land == 0:
        return order[:0]
    while n_land < min(order.size, n_parts) and capacity[order[:n_land]].sum() < n_parts:
        n_land += 1
    return order[:n_land]


def _apportion(shares: np.ndarray, capacity: np.ndarray, n_parts: int) -> np.ndarray:
    """Parts per land mass: one each, then one at a time to the mass whose parts
    are the heaviest, so that the heaviest part is as light as can be.  No mass
    gets more parts than its ``capacity`` (the nodes METIS can place); when the
    capacities run out, fewer than ``n_parts`` are given."""
    count = np.minimum(np.ones(shares.size, dtype=np.int64), capacity)
    while count.sum() < n_parts:
        load = np.where(count < capacity, shares / np.maximum(count, 1), -np.inf)
        if not np.isfinite(load).any():
            break
        count[int(np.argmax(load))] += 1
    return count


def _partition_components(
    linked: PartitionGraph,
    ground: PartitionGraph,
    n_parts: int,
    seed: int,
    imbalance_target: float,
    refine: bool,
    group: np.ndarray | None = None,
) -> np.ndarray:
    """Equal parts of a graph that may have several components, every part one
    piece of land: contiguity first, balance second.

    ``linked`` decides what touches (the ground with the banks of opened
    mainstems joined); ``ground`` holds only shared raster boundaries and is
    used for the balance refinement.  The largest components, of at least half
    an equal share (at most ``n_parts``), are land; each other component (an
    island) is tied to its nearest land node, and a land component with its
    islands is a mass.  The masses get the parts so that the heaviest part is
    as light as can be (``_apportion``), and contiguous METIS cuts each mass
    into equal parts, on its nodes grouped by ``group`` (small tributaries with
    a larger neighbour; each node its own group when None) with its islands
    tied on.  When no component is land (an archipelago), METIS divides the
    components, each linked to its nearest.  Nodes must have cells and a
    centroid.
    """
    n = linked.size
    parts = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return parts
    if group is None:
        group = np.arange(n, dtype=np.int64)
    component = _components(linked)
    component_weight = np.bincount(component, weights=linked.weights)
    # the groups (coarse nodes) of every component: the most parts it can take
    pairs = np.unique(np.column_stack((component, group)), axis=0)
    capacity = np.bincount(pairs[:, 0], minlength=component_weight.size)
    land = _choose_land(component_weight, capacity, n_parts)
    if land.size == 0:
        parts = _archipelago(
            linked, component, component_weight, n_parts, seed, imbalance_target
        )
        return _balance(linked, ground, parts, n_parts, imbalance_target, refine)
    ties, mass_of = _masses(linked, component, land)
    mass = mass_of[component]
    mass_weight = np.bincount(mass, weights=linked.weights, minlength=component_weight.size)
    pairs = np.unique(np.column_stack((mass, group)), axis=0)
    mass_capacity = np.bincount(pairs[:, 0], minlength=component_weight.size)
    counts = _apportion(
        mass_weight[land] / mass_weight.sum() * n_parts, mass_capacity[land], n_parts
    )
    tied = PartitionGraph(
        linked.weights,
        linked.rows,
        linked.cols,
        *merge_edges(
            [linked.edges, ties],
            n,
            [linked.edge_weights, np.ones(ties.shape[0], dtype=np.int64)],
        ),
    )
    first = 0
    for m, k in zip(land, counts):
        nodes = np.flatnonzero(mass == m)
        targets = np.full(k, 1.0 / k)
        parts[nodes] = first + _carve(tied, nodes, group, targets, seed, imbalance_target)
        first += k
    return _balance(linked, ground, parts, n_parts, imbalance_target, refine)


def _balance(
    linked: PartitionGraph,
    ground: PartitionGraph,
    parts: np.ndarray,
    n_parts: int,
    imbalance_target: float,
    refine: bool,
) -> np.ndarray:
    """The refinement and the fragments' absorption of ``_partition_components``."""
    if refine and n_parts > 1:
        targets = np.full(n_parts, 1.0 / n_parts)
        _refine(ground, parts, n_parts, targets, imbalance_target, linked)
        _absorb_fragments(linked, parts, FRAGMENT_SHARE * linked.weights.sum() / n_parts)
    return parts


def _attach_enclaves(
    graph: PartitionGraph, parts: np.ndarray, rest: np.ndarray, n_rest_parts: int
) -> None:
    """Components of the other basins that the assigned (dominant) basins cut off
    from the land -- enclaves, strips along their edge -- and that get no part of
    their own join the part they share the longest boundary with, so that every
    part stays connected on the ground.  Nothing moves in an archipelago.
    """
    sub = _subgraph(graph, rest)
    component = _components(sub)
    component_weight = np.bincount(component, weights=sub.weights)
    capacity = np.bincount(component, minlength=component_weight.size)
    land = _choose_land(component_weight, capacity, n_rest_parts)
    if land.size == 0:
        return
    small = np.ones(component_weight.size, dtype=np.bool_)
    small[land] = False
    assigned = parts >= 0
    first, second = graph.edges[:, 0], graph.edges[:, 1]
    forward = assigned[first] & ~assigned[second]
    backward = assigned[second] & ~assigned[first]
    outside = np.concatenate((second[forward], first[backward]))
    inside = np.concatenate((first[forward], second[backward]))
    length = np.concatenate((graph.edge_weights[forward], graph.edge_weights[backward]))
    local = np.full(graph.size, -1, dtype=np.int64)
    local[rest] = np.arange(rest.size)
    touching = component[local[outside]]
    keep = small[touching]
    if not keep.any():
        return
    # the boundary each such component shares with each part, the longest wins
    n_parts = int(parts.max()) + 1
    shared = np.zeros((component_weight.size, n_parts), dtype=np.int64)
    np.add.at(shared, (touching[keep], parts[inside[keep]]), length[keep])
    cut_off = np.flatnonzero(shared.sum(axis=1) > 0)
    part_of = np.argmax(shared[cut_off], axis=1)
    target = np.full(component_weight.size, -1, dtype=np.int64)
    target[cut_off] = part_of
    moved = target[component] >= 0
    parts[rest[moved]] = target[component[moved]]


def assign_basins(
    graph: PartitionGraph,
    n_parts: int,
    *,
    seed: int = 42,
    imbalance_target: float = 1.005,
    refine: bool = True,
) -> np.ndarray:
    """Method 1: the part of every basin node.

    A basin heavier than an equal share is a part of its own (the largest
    first, at most ``n_parts - 1``), together with the basins it cuts off from
    the others that get no part of their own; the other basins share the other
    parts.  A node without cells or a centroid is merged into one it touches.
    """
    n = graph.size
    if n == 0:
        return np.empty(0, dtype=np.int32)
    merged = _live_nodes(graph)
    if merged is not None:
        live, index = merged
        sub = (
            assign_basins(
                _merge_into(graph, index, live),
                n_parts,
                seed=seed,
                imbalance_target=imbalance_target,
                refine=refine,
            )
            if live.size
            else np.empty(0, dtype=np.int32)
        )
        return _expand(index, sub)
    if n_parts == 1:
        return np.zeros(n, dtype=np.int32)
    if n < n_parts:
        return np.arange(n, dtype=np.int32)
    total = float(graph.weights.sum())
    order = np.argsort(-graph.weights, kind="stable")
    dominant = [
        int(node)
        for node in order[: n_parts - 1]
        if graph.weights[node] > total / n_parts
    ]
    parts = np.full(n, -1, dtype=np.int32)
    for index, node in enumerate(dominant):
        parts[node] = index
    rest = np.flatnonzero(parts < 0)
    if dominant:
        _attach_enclaves(graph, parts, rest, n_parts - len(dominant))
        rest = np.flatnonzero(parts < 0)
    sub = _subgraph(graph, rest)
    parts[rest] = len(dominant) + _partition_components(
        sub, sub, n_parts - len(dominant), seed, imbalance_target, refine
    )
    return parts


def _coarsen_tributaries(
    graph: PartitionGraph,
    is_tributary: np.ndarray,
    min_subtree_size: int,
    min_groups: int = 0,
) -> tuple[np.ndarray, int]:
    """Group small tributaries with the nearest eligible tributary (graph hops).

    Returns the group of every node and the number of groups; a basin node is a
    group of its own, an eligible tributary (at least ``min_subtree_size``, or
    one of the ``min_groups`` heaviest when fewer are that large) starts one,
    and a smaller tributary joins the eligible one it reaches first through
    tributary-to-tributary edges (ties by the lower group).  When no tributary
    is eligible, every tributary is a group of its own.
    """
    n = graph.size
    tributary_weights = graph.weights[is_tributary]
    threshold = min_subtree_size
    if 0 < min_groups < tributary_weights.size:
        heaviest = np.partition(tributary_weights, tributary_weights.size - min_groups)
        threshold = min(threshold, int(heaviest[tributary_weights.size - min_groups]))
    elif min_groups >= tributary_weights.size:
        threshold = 0
    eligible = np.flatnonzero(is_tributary & (graph.weights >= threshold))
    group = np.full(n, -1, dtype=np.int64)
    non_tributary = np.flatnonzero(~is_tributary)
    group[non_tributary] = np.arange(non_tributary.size)
    next_group = non_tributary.size
    tributaries = np.flatnonzero(is_tributary)
    if eligible.size == 0:
        group[tributaries] = next_group + np.arange(tributaries.size)
        return group, next_group + tributaries.size
    xadj, adjncy, _ = _csr(n, graph.edges, graph.edge_weights)
    queue: list[tuple[int, int, int]] = []
    for offset, node in enumerate(eligible):
        group[node] = next_group + offset
        heappush(queue, (0, next_group + offset, int(node)))
    while queue:
        hops, owner, node = heappop(queue)
        if group[node] != owner:
            continue
        for neighbour in adjncy[xadj[node] : xadj[node + 1]]:
            if is_tributary[neighbour] and group[neighbour] < 0:
                group[neighbour] = owner
                heappush(queue, (hops + 1, owner, int(neighbour)))
    unreached = np.flatnonzero(is_tributary & (group < 0))
    count = next_group + eligible.size
    group[unreached] = count + np.arange(unreached.size)
    return group, count + unreached.size


def _align_labels(parts: np.ndarray, weights: np.ndarray, hint: np.ndarray, n_parts: int):
    """Relabel the parts to overlap ``hint`` (another assignment) the most."""
    if n_parts > 8:
        return parts
    overlap = np.zeros((n_parts, n_parts))
    np.add.at(overlap, (parts, hint), weights)
    best = max(
        permutations(range(n_parts)),
        key=lambda mapping: sum(overlap[part, mapping[part]] for part in range(n_parts)),
    )
    return np.asarray(best, dtype=np.int32)[parts]


def _split_stem(
    mine: np.ndarray, tributary_parts: np.ndarray, tributary_position: np.ndarray
) -> tuple[int, int]:
    """Trunk and ``P_min`` of one opened mainstem; ``P_min`` is -1 when every
    tributary is in the trunk.  ``mine`` holds its tributaries, source first."""
    trunk = int(tributary_parts[mine[0]])
    transferred = mine[tributary_parts[mine] != trunk]
    if transferred.size == 0:
        return trunk, -1
    return trunk, int(tributary_position[transferred].min())


def _stem_tributaries(stem: np.ndarray, position: np.ndarray, n_stems: int):
    """The tributaries of each opened mainstem, source first."""
    tributaries = np.flatnonzero(stem >= 0)
    ordered = tributaries[
        np.lexsort((tributaries, position[tributaries], stem[tributaries]))
    ]
    bounds = np.searchsorted(stem[ordered], np.arange(n_stems + 1))
    return [ordered[bounds[k] : bounds[k + 1]] for k in range(n_stems)]


def _mainstem_weights(
    stem: np.ndarray, position: np.ndarray, upto: np.ndarray
) -> np.ndarray:
    """The first ``upto`` cells of each opened mainstem, given to the tributaries
    that enter them.

    A mainstem cell weighs with the first tributary (lowest node) entering it, or
    else with the nearest one entering upstream (the most upstream tributary
    for the cells above it).  Up to ``P_min`` every tributary is in the trunk, and
    so the trunk holds those cells.
    """
    extra = np.zeros(stem.size, dtype=np.int64)
    for k, mine in enumerate(_stem_tributaries(stem, position, len(upto))):
        if mine.size == 0:
            continue
        entered, first = np.unique(position[mine], return_index=True)
        cells = np.arange(int(upto[k]))
        receiver = np.maximum(np.searchsorted(entered, cells, side="right") - 1, 0)
        np.add.at(extra, mine[first[receiver]], 1)
    return extra


def _p_mins(
    parts: np.ndarray, stem: np.ndarray, position: np.ndarray, stem_lengths: np.ndarray
) -> np.ndarray:
    """``P_min`` of each opened mainstem; its length when no other part enters it."""
    p_min = np.asarray(stem_lengths, dtype=np.int64).copy()
    for k, mine in enumerate(_stem_tributaries(stem, position, len(stem_lengths))):
        if mine.size:
            _, first_other = _split_stem(mine, parts, position)
            if first_other >= 0:
                p_min[k] = first_other
    return p_min


def assign_subbasins(
    graph: PartitionGraph,
    is_tributary: np.ndarray,
    position: np.ndarray,
    n_parts: int,
    *,
    stem: np.ndarray | None = None,
    stem_lengths: np.ndarray | None = None,
    hint: np.ndarray | None = None,
    min_subtree_size: int = 100_000,
    seed: int = 42,
    imbalance_target: float = 1.005,
    refine: bool = True,
) -> np.ndarray:
    """Method 2: the part of every node, the opened basins' tributaries included.

    ``is_tributary`` marks the tributary subtrees that drain into an opened
    basin's mainstem, ``stem`` says which mainstem (0 when there is one) and
    ``position`` gives the mainstem cell each enters (0 at the source);
    tributaries entering consecutive cells of one mainstem touch, across it.
    Small tributaries move with the nearest eligible one.  ``hint`` (one part per
    node, Method 1's) only fixes the part labels.  A node without cells or a
    centroid is merged into one it touches.

    With ``stem_lengths`` (cells of each mainstem), the mainstem cells weigh
    with the tributaries that enter them: all of them for METIS, then, while
    the refinement moves ``P_min``, only those above it -- the trunk's -- so that
    the balance is that of the parts as the plan builds them.
    """
    is_tributary = np.asarray(is_tributary, dtype=np.bool_)
    position = np.asarray(position, dtype=np.int64)
    if stem is None:
        stem = np.where(is_tributary, 0, -1)
    stem = np.where(is_tributary, np.asarray(stem, dtype=np.int64), -1)
    if hint is not None:
        hint = np.asarray(hint)
    # the two banks of an opened mainstem touch: tributaries entering consecutive cells
    tributaries = np.flatnonzero(is_tributary)
    ordered = tributaries[np.lexsort((tributaries, position[tributaries], stem[tributaries]))]
    same_stem = stem[ordered[:-1]] == stem[ordered[1:]]
    stem_edges = np.column_stack((ordered[:-1], ordered[1:]))[same_stem]
    edges, edge_weights = merge_edges(
        [graph.edges, stem_edges],
        graph.size,
        [graph.edge_weights, np.ones(stem_edges.shape[0], dtype=np.int64)],
    )
    linked = PartitionGraph(graph.weights, graph.rows, graph.cols, edges, edge_weights)
    # every tributary keeps its stem and position, merged or not: the mainstem
    # cells it is given and P_min are those of the node it is merged into
    all_stem, all_position = stem, position
    merged = _live_nodes(linked)
    if merged is not None:
        live, index = merged
        if live.size == 0:
            return np.zeros(graph.size, dtype=np.int32)
        graph = _merge_into(graph, index, live)
        linked = _merge_into(linked, index, live)
        is_tributary = is_tributary[live]
        hint = None if hint is None else hint[live]

    def node_weights(upto: np.ndarray) -> np.ndarray:
        extra = _mainstem_weights(all_stem, all_position, upto)
        if merged is not None:
            keep = index >= 0
            extra = np.bincount(index[keep], weights=extra[keep], minlength=live.size)
        return cells + extra.astype(np.int64)

    def p_mins(node_parts: np.ndarray) -> np.ndarray:
        if merged is not None:
            node_parts = _expand(index, node_parts)
        return _p_mins(node_parts, all_stem, all_position, stem_lengths)

    cells = graph.weights
    weights = cells
    if stem_lengths is not None:
        stem_lengths = np.asarray(stem_lengths, dtype=np.int64)
        weights = node_weights(stem_lengths)
    ground = PartitionGraph(weights, graph.rows, graph.cols, graph.edges, graph.edge_weights)
    banks = PartitionGraph(weights, linked.rows, linked.cols, linked.edges, linked.edge_weights)
    group, _ = _coarsen_tributaries(
        banks, is_tributary, min_subtree_size, METIS_GROUPS_PER_PART * n_parts
    )
    parts = _partition_components(
        banks, ground, n_parts, seed, imbalance_target, refine, group
    )
    if hint is not None:
        parts = _align_labels(parts, weights, hint, n_parts)
    if stem_lengths is not None and refine and n_parts > 1:
        targets = np.full(n_parts, 1.0 / n_parts)
        p_min = None
        for _ in range(P_MIN_ROUNDS):
            moved = p_mins(parts)
            if p_min is not None and np.array_equal(moved, p_min):
                break
            p_min = moved
            weights = node_weights(p_min)
            ground = PartitionGraph(
                weights, graph.rows, graph.cols, graph.edges, graph.edge_weights
            )
            banks = PartitionGraph(
                weights, linked.rows, linked.cols, linked.edges, linked.edge_weights
            )
            _refine(ground, parts, n_parts, targets, imbalance_target, banks)
            _absorb_fragments(banks, parts, FRAGMENT_SHARE * weights.sum() / n_parts)
    return parts if merged is None else _expand(index, parts)


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------


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


def _basin_partition(
    flw: "Flwdir",
    n_parts: int,
    seed: int,
    refine: bool,
    imbalance_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Method 1 on a raster: parts and loads of the cells, basin ids, labels, node parts."""
    shape = _raster_shape(flw)
    mask = flw.mask.ravel()
    basin_ids = flw.basins().ravel()
    valid = mask & (basin_ids > 0)
    labels = np.unique(basin_ids[valid]).astype(np.int64)
    parts = np.full(flw.size, -1, dtype=np.int32)
    if labels.size == 0:
        return parts, np.zeros(n_parts, dtype=np.int64), basin_ids, labels, labels
    lookup = np.full(int(labels[-1]) + 1, -1, dtype=np.int64)
    lookup[labels] = np.arange(labels.size)
    nodes = np.full(flw.size, -1, dtype=np.int64)
    nodes[valid] = lookup[basin_ids[valid]]
    graph = graph_from_node_raster(nodes.reshape(shape), labels.size)
    node_parts = assign_basins(
        graph, n_parts, seed=seed, imbalance_target=imbalance_target, refine=refine
    )
    parts[valid] = node_parts[nodes[valid]]
    load = np.bincount(parts[parts >= 0], minlength=n_parts).astype(np.int64)
    return parts, load, basin_ids, labels, node_parts


def _open_basins(
    flw: "Flwdir",
    opened: np.ndarray,
    basin_ids: np.ndarray,
    labels: np.ndarray,
    basin_node_parts: np.ndarray,
    seq_d2u: np.ndarray,
    upstream: np.ndarray,
    min_subtree_size: int,
    imbalance_target: float,
    seed: int,
    refine: bool,
) -> PartitionPlan:
    """Method 2 with the basins of node numbers ``opened`` opened along their mainstems."""
    shape = _raster_shape(flw)
    opened_ids = labels[opened]
    in_opened = np.isin(basin_ids, opened_ids)
    mainstems = []
    for basin_id in opened_ids:
        members = np.flatnonzero(basin_ids == basin_id)
        pits = members[flw.idxs_ds[members] == members]
        pit = int(pits[0]) if pits.size else int(members[np.argmax(upstream[members])])
        mainstems.append(_trace_mainstem(flw, pit, upstream, basin_ids == basin_id))
    on_mainstem = np.zeros(flw.size, dtype=np.bool_)
    stem_of_cell = np.full(flw.size, -1, dtype=np.int64)
    position_of_cell = np.full(flw.size, -1, dtype=np.int64)
    for k, mainstem in enumerate(mainstems):
        on_mainstem[mainstem] = True
        stem_of_cell[mainstem] = k
        position_of_cell[mainstem] = np.arange(mainstem.size, dtype=np.int64)
    roots = _label_tributaries(
        flw.idxs_ds, seq_d2u, np.ascontiguousarray(in_opened), on_mainstem
    )
    tributary_roots = np.unique(roots[roots >= 0])

    # nodes: the other basins, then one per tributary; mainstem cells belong to none
    others = np.flatnonzero(~np.isin(np.arange(labels.size), opened))
    basin_lookup = np.full(int(labels[-1]) + 1, -1, dtype=np.int64)
    basin_lookup[labels[others]] = np.arange(others.size)
    nodes = np.full(flw.size, -1, dtype=np.int64)
    in_basin = (basin_ids > 0) & flw.mask.ravel() & ~in_opened
    nodes[in_basin] = basin_lookup[basin_ids[in_basin]]
    tributary_cells = roots >= 0
    nodes[tributary_cells] = others.size + np.searchsorted(
        tributary_roots, roots[tributary_cells]
    )
    n_nodes = others.size + tributary_roots.size
    graph = graph_from_node_raster(nodes.reshape(shape), n_nodes)
    inlets = flw.idxs_ds[tributary_roots]
    is_tributary = np.zeros(n_nodes, dtype=np.bool_)
    is_tributary[others.size :] = True
    stem = np.full(n_nodes, -1, dtype=np.int64)
    stem[others.size :] = stem_of_cell[inlets]
    position = np.full(n_nodes, -1, dtype=np.int64)
    position[others.size :] = position_of_cell[inlets]
    lookup_all = np.full(int(labels[-1]) + 1, -1, dtype=np.int64)
    lookup_all[labels] = np.arange(labels.size)
    hint = np.empty(n_nodes, dtype=np.int64)
    hint[: others.size] = basin_node_parts[others]
    hint[others.size :] = basin_node_parts[lookup_all[basin_ids[tributary_roots]]]
    node_parts = assign_subbasins(
        graph,
        is_tributary,
        position,
        N_TRUNKS,
        stem=stem,
        stem_lengths=np.array([mainstem.size for mainstem in mainstems]),
        hint=hint,
        min_subtree_size=min_subtree_size,
        seed=seed,
        imbalance_target=imbalance_target,
        refine=refine,
    )

    # per opened basin: the trunk holds its most upstream tributary; P_min is where
    # the first other part enters, and the mainstem above it stays with the trunk
    # (tributary nodes are numbered in root order, which breaks ties in position)
    tributary_parts = node_parts[others.size :]
    tributary_position = position[others.size :]
    sizes = graph.weights[others.size :]
    stem_tributaries = _stem_tributaries(
        stem[others.size :], tributary_position, len(mainstems)
    )
    refined = np.full(flw.size, -1, dtype=np.int32)
    valid = nodes >= 0
    refined[valid] = node_parts[nodes[valid]]
    stems = []
    for k, mainstem in enumerate(mainstems):
        mine = stem_tributaries[k]
        if mine.size == 0:
            # a basin that is its mainstem alone: whole, in its Method 1 part
            refined[mainstem] = int(basin_node_parts[opened[k]])
            continue
        trunk, p_min = _split_stem(mine, tributary_parts, tributary_position)
        if p_min < 0:
            refined[mainstem] = trunk
            continue
        refined[mainstem[:p_min]] = trunk
        refined[mainstem[p_min:]] = MAINSTEM
        cut = mine[tributary_position[mine] >= p_min]
        cut_outlets = tributary_roots[cut].astype(flw.idxs_ds.dtype)
        predecessor = int(mainstem[p_min - 1]) if p_min > 0 else -1
        basin_size = int(np.count_nonzero(basin_ids == opened_ids[k]))
        accounted = mainstem.size - p_min + int(sizes[cut].sum())
        if predecessor >= 0:
            accounted += int(upstream[predecessor])
        if accounted != basin_size:
            raise RuntimeError(
                "Subbasin partition does not account for every opened-basin cell: "
                f"{accounted} != {basin_size}."
            )
        stems.append(
            FifthRegion(
                mainstem=mainstem[p_min:],
                predecessor=predecessor,
                cut_outlets=cut_outlets,
                cut_inlets=flw.idxs_ds[cut_outlets].astype(flw.idxs_ds.dtype),
                cut_ranks=tributary_parts[cut].astype(np.int8),
                trunk=trunk,
            )
        )
    final_load = np.bincount(
        refined[(refined >= 0) & (refined < N_TRUNKS)], minlength=N_TRUNKS
    ).astype(np.int64)
    return PartitionPlan(
        parts=refined.reshape(flw.shape),
        loads=final_load,
        basin_ids=basin_ids.reshape(flw.shape),
        level="subbasin",
        stems=tuple(stems),
    )


def _subbasin_partition(
    flw: "Flwdir",
    parts: np.ndarray,
    load: np.ndarray,
    basin_ids: np.ndarray,
    labels: np.ndarray,
    basin_node_parts: np.ndarray,
    min_subtree_size: int,
    imbalance_target: float,
    seed: int,
    refine: bool,
) -> PartitionPlan:
    """Method 2 on a raster: open the largest basins while the parts are unequal.

    Method 1's parts stand when they are within ``imbalance_target``; otherwise
    the largest basin is opened, then the next, and the most balanced plan
    (Method 1's included) is kept.
    """
    method_1 = _empty_plan(flw, parts, load, basin_ids, "subbasin")
    best_ratio = float(load.max() / max(load.mean(), 1))
    if labels.size == 0 or best_ratio <= imbalance_target:
        return method_1
    valid = parts >= 0
    weights = np.bincount(np.searchsorted(labels, basin_ids[valid]), minlength=labels.size)
    order = np.argsort(-weights, kind="stable")
    seq_d2u = core.idxs_seq_dfs(flw.idxs_ds, flw.idxs_pit, flw._mv)
    upstream = streams.accuflux(
        flw.idxs_ds, seq_d2u, np.ones(flw.size, dtype=np.int64), -1
    )
    best = method_1
    for n_opened in range(1, min(MAX_OPENED_BASINS, labels.size) + 1):
        plan = _open_basins(
            flw,
            order[:n_opened],
            basin_ids,
            labels,
            basin_node_parts,
            seq_d2u,
            upstream,
            min_subtree_size,
            imbalance_target,
            seed,
            refine,
        )
        ratio = float(plan.loads.max() / max(plan.loads.mean(), 1))
        if ratio < best_ratio - 1e-12:
            best, best_ratio = plan, ratio
        if ratio <= imbalance_target:
            break
    return best


def _empty_plan(
    flw: "Flwdir",
    parts: np.ndarray,
    load: np.ndarray,
    basin_ids: np.ndarray,
    level: str,
) -> PartitionPlan:
    return PartitionPlan(
        parts=parts.reshape(flw.shape),
        loads=load,
        basin_ids=basin_ids.reshape(flw.shape),
        level=level,
    )


def partition_plan(
    flw: "Flwdir",
    level: Literal["basin", "subbasin"] = "subbasin",
    n_parts: int = N_TRUNKS,
    *,
    seed: int = 42,
    refine: bool = True,
    min_subtree_size: int = 100_000,
    imbalance_target: float = 1.005,
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

    parts, load, basin_ids, labels, node_parts = _basin_partition(
        flw, n_parts, seed, refine, imbalance_target
    )
    if level == "basin":
        return _empty_plan(flw, parts, load, basin_ids, level)
    return _subbasin_partition(
        flw,
        parts,
        load,
        basin_ids,
        labels,
        node_parts,
        min_subtree_size,
        imbalance_target,
        seed,
        refine,
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
