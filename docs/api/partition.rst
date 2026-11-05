.. currentmodule:: pyflwdir

Partition methods
-----------------

A partition divides a raster into parts, for upstream accumulation in processes. It
is an optional feature: install it with ``pip install pyflwdir[partition]``.

Use :py:meth:`FlwdirRaster.partition` when you only need the part of each cell and
the cell count of each part, and :func:`partition.partition_plan` when you also want
to inspect the opened mainstem sections, which the accumulation finishes last.

A partition only says which part each cell belongs to. Upstream accumulation is the
only computation that uses one: :py:meth:`FlwdirRaster.accuflux` and
:py:meth:`FlwdirRaster.upstream_area` with ``parallel=True`` and ``n_processes``
above one, which build their own and do not take a plan. Building a plan does not
make any other method run in parallel.

A tributary subtree is never cut, so the heaviest part is at least as heavy as the
largest subtree of an opened basin, however many parts are asked.

.. autosummary::
   :toctree: ../_generated

   partition.partition_plan
   partition.PartitionPlan
   partition.MainstemSection
