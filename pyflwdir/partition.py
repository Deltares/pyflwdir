"""Split flow networks into independent regions for process-level parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numba import njit

from . import basins, core, streams

if TYPE_CHECKING:
    from .flwdir import Flwdir

MAINSTEM = -2
"""Cells held back for the second stage of a subbasin partition."""


def _lpt(weights: np.ndarray, n_parts: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign largest items first to the currently lightest partition."""
    weights = np.asarray(weights, dtype=np.float64)
    load = np.zeros(n_parts, dtype=np.float64)
    assignment = np.empty(weights.size, dtype=np.int32)
    for item in np.argsort(-weights):
        lightest = int(np.argmin(load))
        assignment[item] = lightest
        load[lightest] += weights[item]
    return assignment, load


def _mainstem(
    idxs_ds: np.ndarray,
    pit: int,
    upstream: np.ndarray,
    indptr: np.ndarray,
    idxs_us: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Walk from a pit to a headwater, taking the largest tributary."""
    stem = [pit]
    seen = {pit}
    cell = pit
    while True:
        donors = idxs_us[indptr[cell] : indptr[cell + 1]]
        donors = donors[mask[donors]]
        if donors.size == 0:
            break
        cell = int(donors[np.argmax(upstream[donors])])
        if cell in seen:
            break
        seen.add(cell)
        stem.append(cell)
    return np.asarray(stem, dtype=idxs_ds.dtype)


@njit(cache=True)
def _label_subtrees(
    idxs_ds: np.ndarray,
    seq: np.ndarray,
    on_stem: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Label each tributary cell by the root where it leaves the mainstem."""
    roots = np.full(idxs_ds.size, -1, dtype=np.int64)
    for idx in seq:
        if eligible[idx] == 0 or on_stem[idx] != 0:
            continue
        idx_ds = idxs_ds[idx]
        roots[idx] = idx if on_stem[idx_ds] != 0 else roots[idx_ds]
    return roots


def partition(
    flw: "Flwdir",
    n_parts: int = 4,
    level: Literal["basin", "subbasin"] = "subbasin",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign valid cells to independent process-level regions.

    Whole basins are assigned with longest-processing-time-first when
    ``level="basin"``. With ``level="subbasin"``, a basin larger than one
    partition's target load is cut along its mainstem; tributary subtrees are
    assigned independently and the mainstem is marked :data:`MAINSTEM` for a
    second stage after all regions finish.
    """
    if level not in ("basin", "subbasin"):
        raise ValueError("level must be 'basin' or 'subbasin'")
    if n_parts < 1:
        raise ValueError("n_parts must be at least 1")

    idxs_ds = flw.idxs_ds
    mask = flw.mask.ravel()
    parts = np.full(idxs_ds.size, -1, dtype=np.int32)
    if not np.any(mask):
        return parts.reshape(flw.shape), np.zeros(n_parts, dtype=np.float64)

    basin_ids = basins.basins(idxs_ds, flw.idxs_pit, flw.idxs_seq)
    counts = np.bincount(basin_ids[mask])
    labels = np.flatnonzero(counts)
    labels = labels[labels != 0]
    sizes = counts[labels]

    if labels.size == 0:
        parts[mask] = 0
        return parts.reshape(flw.shape), np.zeros(n_parts, dtype=np.float64)

    def whole_basins() -> tuple[np.ndarray, np.ndarray]:
        assignment, load = _lpt(sizes, n_parts)
        lookup = np.full(counts.size, -1, dtype=np.int32)
        lookup[labels] = assignment
        parts[mask] = lookup[basin_ids[mask]]
        stranded = mask & (parts == -1)
        if np.any(stranded):
            parts[stranded] = int(np.argmin(load))
        return parts.reshape(flw.shape), load

    if level == "basin":
        return whole_basins()

    target = sizes.sum() / float(n_parts)
    large = labels[(sizes > target) & (sizes >= 3)]
    if large.size == 0:
        return whole_basins()

    seq = flw.idxs_seq
    upstream = streams.accuflux(
        idxs_ds,
        seq,
        np.ones(idxs_ds.size, dtype=np.float64),
        -1,
    )
    indptr, idxs_us = core.upstream_csr(idxs_ds, flw._mv)
    decomposed = np.isin(basin_ids, large) & mask
    on_stem = np.zeros(idxs_ds.size, dtype=np.uint8)
    for basin_id in large:
        members = np.flatnonzero(basin_ids == basin_id)
        pits = members[idxs_ds[members] == members]
        pit = (
            int(pits[0])
            if pits.size == 1
            else int(members[np.argmax(upstream[members])])
        )
        on_stem[_mainstem(idxs_ds, pit, upstream, indptr, idxs_us, mask)] = 1

    parts[on_stem != 0] = MAINSTEM
    roots = _label_subtrees(
        idxs_ds,
        seq,
        on_stem,
        np.ascontiguousarray(decomposed, dtype=np.uint8),
    )

    whole = labels[~np.isin(labels, large)]
    weights = list(counts[whole])
    members: list[tuple[str, int | np.ndarray]] = [
        ("basin", int(label)) for label in whole
    ]

    labelled = np.flatnonzero(roots >= 0)
    if labelled.size:
        keys = roots[labelled]
        order = np.argsort(keys, kind="stable")
        labelled = labelled[order]
        keys = keys[order]
        _, starts = np.unique(keys, return_index=True)
        starts = np.append(starts, labelled.size)
        for pos in range(starts.size - 1):
            group = labelled[starts[pos] : starts[pos + 1]]
            weights.append(group.size)
            members.append(("cells", group))

    assignment, load = _lpt(np.asarray(weights), n_parts)
    lookup = np.full(counts.size, -1, dtype=np.int32)
    for item, (kind, value) in enumerate(members):
        if kind == "basin":
            lookup[int(value)] = assignment[item]
        else:
            parts[value] = assignment[item]

    keep = mask & ~decomposed
    parts[keep] = lookup[basin_ids[keep]]
    stranded = mask & (parts == -1)
    if np.any(stranded):
        parts[stranded] = int(np.argmin(load))
    return parts.reshape(flw.shape), load
