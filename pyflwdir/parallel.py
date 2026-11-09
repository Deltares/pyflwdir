"""Upstream accumulation in processes, with threads inside each process.

The raster is divided into parts by :py:meth:`pyflwdir.FlwdirRaster.partition`, which
returns a :class:`~pyflwdir.partition.PartitionPlan`.  Given that plan, the
accumulation runs in two stages:

1. Each part is accumulated in its own process, one layer at a time in threads, and
   stops at the boundary of the part.
2. When every process has finished, the parent process accumulates the mainstem
   sections, each from ``P_min`` down to the pit, with what the parts left at their
   boundaries.

Before it starts, the plan is checked against the flow directions: a plan that does
not fit them, for example one made before they changed, would otherwise give a wrong
result without an error.

Upstream accumulation is all that runs in parallel (``accuflux`` with
``direction="up"`` and the raster ``upstream_area``): no other method uses the
layers or the parts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_all_start_methods, get_context, shared_memory
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numba import config, get_num_threads, njit, set_num_threads

from . import partition, streams

if TYPE_CHECKING:
    from .flwdir import Flwdir

ArraySpec: TypeAlias = tuple[str, tuple[int, ...], str]
Section: TypeAlias = tuple[np.ndarray, int, np.ndarray, np.ndarray]
"""What stage 2 reads of a mainstem section: ``mainstem``, ``predecessor``,
``cut_outlets`` and ``cut_inlets``."""


# ---------------------------------------------------------------------------
# Checking a plan against the flow directions
# ---------------------------------------------------------------------------


def _does_not_fit(problem: str) -> ValueError:
    return ValueError(
        f"The partition plan does not fit these flow directions: {problem}. "
        "Was it made for another raster, or before the flow directions changed "
        "(for example by add_pits)? Make a new one with flw.partition()."
    )


def _cell_indices(values: Any, n_cells: int, what: str) -> np.ndarray:
    """``values`` as a 1-D array of cell indices, each within the raster."""
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise _does_not_fit(f"{what} is not a 1-D array of cell indices")
    if array.size and (array.min() < 0 or array.max() >= n_cells):
        raise _does_not_fit(f"{what} holds a cell index outside the raster")
    return array.astype(np.intp, copy=False)


_OFF_NETWORK_WITH_A_PART = 1
_DRAINS_WITHOUT_A_PART = 2
_DRAINS_INTO_ANOTHER_PART = 3


@njit(cache=True)
def _check_cells(idxs_ds, parts, mv, n_parts, layered_cells):
    """Check the part of every cell in one pass, as :py:func:`_check_plan` describes.

    ``layered_cells`` are the cells in the layers: those that reach a pit. The others
    take part in no accumulation, so their part is not checked.

    Returns 0, or the first problem found and its cell, and the number of hand-overs:
    cells of a part that drain into a mainstem section.
    """
    reaches_pit = np.zeros(idxs_ds.size, dtype=np.bool_)
    for cell in layered_cells:
        reaches_pit[cell] = True
    n_handovers = 0
    for cell in range(idxs_ds.size):
        down = idxs_ds[cell]
        part = parts[cell]
        if down == mv:
            if part != -1:
                return _OFF_NETWORK_WITH_A_PART, cell, n_handovers
        elif down != cell and reaches_pit[cell]:
            if part == -1:
                return _DRAINS_WITHOUT_A_PART, cell, n_handovers
            if part < n_parts:
                if parts[down] == n_parts:
                    n_handovers += 1
                elif parts[down] != part:
                    return _DRAINS_INTO_ANOTHER_PART, cell, n_handovers
    return 0, -1, n_handovers


def _handovers(idxs_ds, parts, mv, n_parts):
    """The cells of a part that drain into a mainstem section."""
    drains = (idxs_ds != mv) & (idxs_ds != np.arange(idxs_ds.size))
    sources = np.flatnonzero(drains & (parts >= 0) & (parts < n_parts))
    return sources[parts[idxs_ds[sources]] == n_parts]


def _check_plan(
    flw: "Flwdir", plan: Any, layered_cells: np.ndarray
) -> tuple[np.ndarray, int, list[Section]]:
    """Check that ``plan`` fits the flow directions of ``flw``.

    The two stages give the serial result when, and only when:

    * every cell of a part is a pit, or drains into a cell of the same part, or
      drains into a mainstem section (a hand-over);
    * every mainstem section is a chain of cells, each draining into the next,
      down to a pit, and each cell labelled as a section is in one section;
    * every hand-over is listed once, as the ``predecessor`` of the section it
      drains into or as one of its ``cut_outlets``, and the ``cut_inlets`` follow
      the section from upstream down.

    Cells that never reach a pit, in or draining into a loop of the flow directions,
    are not in ``layered_cells`` and take part in no accumulation, serial or
    parallel, so their part is not checked.

    Returns the part of each cell as a flat array, the number of parts, and what
    stage 2 reads of each section.
    """
    if not isinstance(plan, partition.PartitionPlan):
        raise TypeError(
            "partition_plan must be a PartitionPlan, made with flw.partition(), "
            f"not {type(plan).__name__}"
        )
    loads = np.asarray(plan.loads)
    if loads.ndim != 1 or loads.size == 0:
        raise _does_not_fit("loads must hold the cell count of at least one part")
    n_parts = int(loads.size)

    parts = np.asarray(plan.parts)
    shape = flw.shape if isinstance(flw.shape, tuple) else (flw.shape,)
    if parts.shape != tuple(shape):
        raise _does_not_fit(
            f"parts has shape {parts.shape}, the flow directions {tuple(shape)}"
        )
    if parts.dtype.kind not in "iu":
        raise _does_not_fit("parts must hold integer labels")
    parts = np.ascontiguousarray(parts.ravel())
    if parts.size and (parts.min() < -1 or parts.max() > n_parts):
        raise _does_not_fit(
            f"parts holds labels outside -1 to {n_parts}; loads has {n_parts} parts, "
            f"so {n_parts} marks a mainstem section"
        )

    idxs_ds = flw.idxs_ds
    n_cells = idxs_ds.size
    problem, cell, n_handovers = _check_cells(
        idxs_ds, parts, int(flw._mv), n_parts, layered_cells
    )
    if problem == _OFF_NETWORK_WITH_A_PART:
        raise _does_not_fit(
            f"cell {cell} is off the river network but has a part; it must be -1"
        )
    if problem == _DRAINS_WITHOUT_A_PART:
        raise _does_not_fit(
            f"cell {cell} has no part (-1) but its water flows on to a pit; only cells "
            "off the network, a pit that nothing drains into, and cells in or draining "
            "into a loop of the flow directions can be without a part"
        )
    if problem == _DRAINS_INTO_ANOTHER_PART:
        raise _does_not_fit(
            f"cell {cell} of part {parts[cell]} drains into cell {idxs_ds[cell]}, "
            f"labelled {parts[idxs_ds[cell]]}; a part can hand its water on only to "
            "a mainstem section"
        )

    # the mainstem sections and the hand-overs they list
    sections: list[Section] = []
    mainstems = []
    listed = []
    for number, stem in enumerate(plan.stems):
        name = f"mainstem section {number}"
        mainstem = _cell_indices(stem.mainstem, n_cells, f"the mainstem of {name}")
        if mainstem.size == 0:
            raise _does_not_fit(f"{name} has no cells")
        if np.any(parts[mainstem] != n_parts):
            raise _does_not_fit(f"{name} holds a cell not labelled {n_parts} in parts")
        if (
            np.any(idxs_ds[mainstem[:-1]] != mainstem[1:])
            or idxs_ds[mainstem[-1]] != mainstem[-1]
        ):
            raise _does_not_fit(
                f"{name} is not a chain of cells, each draining into the next, "
                "down to a pit"
            )

        predecessor = int(stem.predecessor)
        if predecessor != -1:
            if not 0 <= predecessor < n_cells:
                raise _does_not_fit(f"the predecessor of {name} is outside the raster")
            if not 0 <= parts[predecessor] < n_parts:
                raise _does_not_fit(f"the predecessor of {name} is not in a part")
            if idxs_ds[predecessor] != mainstem[0]:
                raise _does_not_fit(
                    f"the predecessor of {name} does not drain into its first cell"
                )
            listed.append(np.array([predecessor], dtype=np.intp))

        cut_outlets = _cell_indices(stem.cut_outlets, n_cells, f"cut_outlets of {name}")
        cut_inlets = _cell_indices(stem.cut_inlets, n_cells, f"cut_inlets of {name}")
        if cut_outlets.size != cut_inlets.size:
            raise _does_not_fit(
                f"{name} has {cut_outlets.size} cut_outlets "
                f"but {cut_inlets.size} cut_inlets"
            )
        if cut_outlets.size:
            cut_labels = parts[cut_outlets]
            if np.any((cut_labels < 0) | (cut_labels >= n_parts)):
                raise _does_not_fit(f"a cut outlet of {name} is not in a part")
            if np.any(idxs_ds[cut_outlets] != cut_inlets):
                raise _does_not_fit(
                    f"a cut outlet of {name} does not drain into its cut inlet"
                )
            order = np.argsort(mainstem, kind="stable")
            found = np.searchsorted(mainstem[order], cut_inlets)
            found[found == mainstem.size] = 0
            if np.any(mainstem[order][found] != cut_inlets):
                raise _does_not_fit(f"a cut inlet of {name} is not on that section")
            if np.any(np.diff(order[found]) < 0):
                raise _does_not_fit(
                    f"the cut inlets of {name} do not follow it from upstream down"
                )
            listed.append(cut_outlets)

        mainstems.append(mainstem)
        sections.append((mainstem, predecessor, cut_outlets, cut_inlets))

    n_section_cells = int(np.count_nonzero(parts == n_parts))
    all_mainstem = np.concatenate(mainstems) if mainstems else np.empty(0, np.intp)
    if (
        all_mainstem.size != n_section_cells
        or np.unique(all_mainstem).size != all_mainstem.size
    ):
        raise _does_not_fit(
            f"the mainstem sections do not hold each of the {n_section_cells} cells "
            f"labelled {n_parts} in parts exactly once"
        )

    listed_cells = np.sort(np.concatenate(listed)) if listed else np.empty(0, np.intp)
    twice = listed_cells[1:][listed_cells[1:] == listed_cells[:-1]]
    if twice.size:
        raise _does_not_fit(
            f"cell {int(twice[0])} is listed twice, as a predecessor or cut outlet, so "
            "its water would be added twice"
        )
    # every listed cell is a hand-over, so a hand-over is missing if the counts differ
    if listed_cells.size != n_handovers:
        missing = np.setdiff1d(
            _handovers(idxs_ds, parts, flw._mv, n_parts), listed_cells
        )
        raise _does_not_fit(
            f"cell {int(missing[0])} drains into a mainstem section but is not its "
            "predecessor or one of its cut outlets, so its water would be lost"
        )
    return parts, n_parts, sections


# ---------------------------------------------------------------------------
# The two stages
# ---------------------------------------------------------------------------


def _create_shared(array: np.ndarray) -> tuple[shared_memory.SharedMemory, ArraySpec]:
    array = np.ascontiguousarray(array)
    memory = shared_memory.SharedMemory(create=True, size=array.nbytes)
    view = np.ndarray(array.shape, dtype=array.dtype, buffer=memory.buf)
    view[...] = array
    return memory, (memory.name, array.shape, array.dtype.str)


def _open_shared(
    spec: ArraySpec,
) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    name, shape, dtype = spec
    memory = shared_memory.SharedMemory(name=name)
    return memory, np.ndarray(shape, dtype=np.dtype(dtype), buffer=memory.buf)


@njit(cache=True)
def _cells_by_part(cells, offsets, parts, n_parts):
    """Group the cells, given layer by layer, by part, each part still layer by layer.

    Returns the grouped cells and, for each part, where each of its layers starts in
    them: the cells of ``part`` in ``layer`` are
    ``grouped[starts[part, layer] : starts[part, layer + 1]]``.
    """
    n_layers = offsets.size - 1
    sizes = np.zeros((n_parts, n_layers), dtype=np.int64)
    for layer in range(n_layers):
        for i in range(offsets[layer], offsets[layer + 1]):
            part = parts[cells[i]]
            if 0 <= part < n_parts:
                sizes[part, layer] += 1
    starts = np.zeros((n_parts, n_layers + 1), dtype=np.int64)
    total = 0
    for part in range(n_parts):
        for layer in range(n_layers):
            starts[part, layer] = total
            total += sizes[part, layer]
        starts[part, n_layers] = total
    grouped = np.empty(total, dtype=cells.dtype)
    filled = starts[:, :n_layers].copy()
    for layer in range(n_layers):
        for i in range(offsets[layer], offsets[layer + 1]):
            part = parts[cells[i]]
            if 0 <= part < n_parts:
                grouped[filled[part, layer]] = cells[i]
                filled[part, layer] += 1
    return grouped, starts


def _divide_work(
    cells: np.ndarray,
    offsets: np.ndarray,
    parts: np.ndarray,
    n_parts: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """For each part with cells, its cells by CFDS layer: what its process runs."""
    grouped, starts = _cells_by_part(cells, offsets, parts, n_parts)
    work = []
    for part in range(n_parts):
        first, end = starts[part, 0], starts[part, -1]
        if end > first:
            work.append((part, grouped[first:end], starts[part] - first))
    return work


def _accuflux_worker(
    idxs_ds_spec: ArraySpec,
    parts_spec: ArraySpec,
    accu_spec: ArraySpec,
    part: int,
    cells: np.ndarray,
    offsets: np.ndarray,
    nodata: float,
    n_threads: int,
) -> int:
    """Stage 1 in one process: accumulate one part in the shared arrays."""
    memories: list[shared_memory.SharedMemory] = []
    try:
        idxs_memory, idxs_ds = _open_shared(idxs_ds_spec)
        parts_memory, parts = _open_shared(parts_spec)
        accu_memory, accu = _open_shared(accu_spec)
        memories.extend((idxs_memory, parts_memory, accu_memory))
        set_num_threads(min(max(1, n_threads), config.NUMBA_NUM_THREADS))
        streams.accuflux_partitioned_push(
            idxs_ds,
            parts,
            part,
            cells,
            offsets,
            accu,
            nodata,
        )
        return part
    finally:
        for memory in memories:
            memory.close()


def _accumulate_parts(
    flw: "Flwdir",
    data: np.ndarray,
    nodata: float,
    parts: np.ndarray,
    work: list[tuple[int, np.ndarray, np.ndarray]],
    threads_per_process: int,
    start_method: str,
) -> np.ndarray:
    """Stage 1: each part in its own process, in shared memory."""
    idxs_memory = parts_memory = accu_memory = None
    try:
        idxs_memory, idxs_spec = _create_shared(flw.idxs_ds)
        parts_memory, parts_spec = _create_shared(parts)
        accu_memory, accu_spec = _create_shared(data)
        try:
            context = get_context(start_method)
            with ProcessPoolExecutor(
                max_workers=len(work),
                mp_context=context,
            ) as executor:
                futures = [
                    executor.submit(
                        _accuflux_worker,
                        idxs_spec,
                        parts_spec,
                        accu_spec,
                        part,
                        part_cells,
                        part_offsets,
                        nodata,
                        threads_per_process,
                    )
                    for part, part_cells, part_offsets in work
                ]
                for future in futures:
                    future.result()
        except BrokenProcessPool as error:
            message = (
                "An accumulation process terminated abruptly. Check "
                "available shared memory and process memory, and inspect the "
                "chained worker error."
            )
            if start_method == "spawn":
                message += (
                    " Also call spawn-based code from an importable script "
                    "under `if __name__ == '__main__':`."
                )
            raise RuntimeError(message) from error
        return np.ndarray(data.shape, dtype=data.dtype, buffer=accu_memory.buf).copy()
    finally:
        for memory in (idxs_memory, parts_memory, accu_memory):
            if memory is not None:
                memory.close()
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass


def accuflux(
    flw: "Flwdir",
    data: np.ndarray,
    nodata: float,
    plan: partition.PartitionPlan,
    threads_per_process: int | None,
    start_method: str,
) -> np.ndarray:
    """Check the arguments and the plan, accumulate the parts in processes, one
    process a part, and then the mainstem sections."""
    if threads_per_process is not None and threads_per_process < 1:
        raise ValueError("threads_per_process must be at least 1")
    if start_method not in get_all_start_methods():
        raise ValueError(
            f"start_method must be one of {get_all_start_methods()}, got {start_method!r}"
        )
    if start_method == "spawn":
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if main_file and (
            str(main_file).startswith("<") or not Path(main_file).exists()
        ):
            raise RuntimeError(
                "start_method='spawn' needs an importable Python script. "
                "Run the call below an `if __name__ == '__main__':` guard, "
                "or choose another supported start method."
            )

    cells, offsets = flw._layer_decomposition("cfds")
    parts, n_parts, sections = _check_plan(flw, plan, cells)
    if threads_per_process is None:
        threads_per_process = max(1, get_num_threads() // n_parts)

    work = _divide_work(cells, offsets, parts, n_parts)
    if work:
        accu = _accumulate_parts(
            flw, data, nodata, parts, work, threads_per_process, start_method
        )
    else:
        accu = data.copy()

    # stage 2: one mainstem section per opened basin; they lie in different basins,
    # so any order
    for mainstem, predecessor, cut_outlets, cut_inlets in sections:
        streams.accuflux_subbasin_mainstem(
            mainstem,
            predecessor,
            cut_outlets,
            cut_inlets,
            data,
            accu,
            nodata,
        )
    return accu
