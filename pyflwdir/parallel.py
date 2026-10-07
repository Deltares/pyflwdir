"""Hybrid process- and thread-parallel flow accumulation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_all_start_methods, get_context, shared_memory
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numba import config, get_num_threads, set_num_threads

from . import streams
from .partition import MAINSTEM

if TYPE_CHECKING:
    from .flwdir import Flwdir

ArraySpec: TypeAlias = tuple[str, tuple[int, ...], str]


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


def _part_decomposition(
    cells: np.ndarray,
    offsets: np.ndarray,
    parts: np.ndarray,
    part: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = parts[cells] == part
    part_cells = np.ascontiguousarray(cells[selected])
    layer_sizes = np.empty(offsets.size - 1, dtype=np.int64)
    for layer in range(layer_sizes.size):
        layer_cells = cells[offsets[layer] : offsets[layer + 1]]
        layer_sizes[layer] = np.count_nonzero(parts[layer_cells] == part)
    part_offsets = np.zeros(offsets.size, dtype=np.int64)
    np.cumsum(layer_sizes, out=part_offsets[1:])
    return part_cells, part_offsets


def _plan(
    flw: "Flwdir", n_processes: int, partition_level: str
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray]], np.ndarray]:
    key = f"hybrid_{partition_level}_{n_processes}"
    if key in flw._cached:
        return flw._cached[key]

    parts, _ = flw.partition(n_parts=n_processes, level=partition_level)
    parts = np.ascontiguousarray(parts.ravel())
    cells, offsets = flw._layer_decomposition("cfds")
    work = []
    for part in range(n_processes):
        part_cells, part_offsets = _part_decomposition(cells, offsets, parts, part)
        if part_cells.size:
            work.append((part, part_cells, part_offsets))

    seq = flw.idxs_seq[::-1]
    receivers = flw.idxs_ds[seq]
    joins_stem = (parts[seq] >= 0) & (parts[receivers] == MAINSTEM)
    on_stem = parts[seq] == MAINSTEM
    moving = receivers != seq
    merge_cells = np.ascontiguousarray(seq[(joins_stem | on_stem) & moving])
    plan = (parts, work, merge_cells)
    if flw.cache:
        flw._cached[key] = plan
    return plan


def accuflux(
    flw: "Flwdir",
    data: np.ndarray,
    nodata: float,
    n_processes: int,
    threads_per_process: int | None,
    partition_level: str,
    start_method: str,
) -> np.ndarray:
    """Accumulate with independent process regions and CFDS threads inside each."""
    if n_processes < 2:
        raise ValueError("n_processes must be at least 2 for hybrid accumulation")
    if threads_per_process is None:
        threads_per_process = max(1, get_num_threads() // n_processes)
    if threads_per_process < 1:
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

    parts, work, merge_cells = _plan(flw, n_processes, partition_level)
    if not work:
        accu = data.copy()
        streams.accuflux_partition_mainstem(flw.idxs_ds, merge_cells, accu, nodata)
        return accu

    idxs_memory = parts_memory = accu_memory = None
    try:
        idxs_memory, idxs_spec = _create_shared(flw.idxs_ds)
        parts_memory, parts_spec = _create_shared(parts)
        accu_memory, accu_spec = _create_shared(data)
        accu = np.ndarray(data.shape, dtype=data.dtype, buffer=accu_memory.buf)

        try:
            context = get_context(start_method)
            with ProcessPoolExecutor(
                max_workers=min(n_processes, len(work)),
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
        except BrokenProcessPool as exc:
            message = (
                "A hybrid accumulation worker terminated abruptly. Check "
                "available shared memory and process memory, and inspect the "
                "chained worker error."
            )
            if start_method == "spawn":
                message += (
                    " Also call spawn-based code from an importable script "
                    "under `if __name__ == '__main__':`."
                )
            raise RuntimeError(message) from exc

        if merge_cells.size:
            streams.accuflux_partition_mainstem(flw.idxs_ds, merge_cells, accu, nodata)
        return accu.copy()
    finally:
        for memory in (idxs_memory, parts_memory, accu_memory):
            if memory is not None:
                memory.close()
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass
