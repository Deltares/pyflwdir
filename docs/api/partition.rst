.. currentmodule:: pyflwdir

Partition methods
-----------------

A partition divides a raster into parts, for upstream accumulation in processes. It
is an optional feature: install it with ``pip install pyflwdir[partition]``.

:py:meth:`FlwdirRaster.partition` divides the raster into one part a process and
returns a :class:`partition.PartitionPlan`: the part of each cell, the cell count of
each part, and the opened mainstem sections, which the accumulation finishes last.
:func:`partition.partition_plan` is the function behind it.

Upstream accumulation is the only computation that uses a plan: pass it as
``partition_plan`` to :py:meth:`FlwdirRaster.accuflux` or
:py:meth:`FlwdirRaster.upstream_area`, with ``parallel=True``. The accumulation
checks the plan against the flow directions first, and one plan serves any number of
calls. Making a plan does not make any other method run in parallel.

A tributary subtree is never cut, so the heaviest part is at least as heavy as the
largest subtree of an opened basin, however many parts are asked.

.. autosummary::
   :toctree: ../_generated

   partition.partition_plan
   partition.PartitionPlan
   partition.MainstemSection
