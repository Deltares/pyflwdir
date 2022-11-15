#############################################################################
PyFlwDir: Fast methods to work with hydro- and topography data in pure Python
#############################################################################

.. image:: https://codecov.io/gh/Deltares/PyFlwDir/branch/main/graph/badge.svg?token=N4VMHJJAV3
    :target: https://codecov.io/gh/Deltares/PyFlwDir

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/PyFlwDir/latest
    :alt: Latest developers docs

.. image:: https://badge.fury.io/py/PyFlwDir.svg
    :target: https://pypi.org/project/PyFlwDir/
    :alt: Latest PyPI version

.. image:: https://anaconda.org/conda-forge/PyFlwDir/badges/version.svg
    :target: https://anaconda.org/conda-forge/PyFlwDir

.. image:: https://zenodo.org/badge/409871473.svg
   :target: https://zenodo.org/badge/latestdoi/409871473

Intro
-----

PyFlwDir contains a series of methods to work with gridded DEM and flow direction 
datasets, which are key to many workflows in many earth sciences. 
PyFlwDir supports several flow direction data conventions and can easily be extended to include more. 
The package contains some unique methods such as Iterative Hydrography Upscaling (IHU) 
method to upscale flow directions from high resolution data to coarser model resolution. 

PyFlwDir is in pure python and powered by `numba <https://numba.pydata.org/>`_ to keep it fast.


Featured methods
----------------

- flow directions from elevation data using a steepest gradient algorithm
- strahler stream order
- flow direction upscaling
- (sub)basin delineation
- pfafstetter subbasins delineation
- classic stream order
- height above nearest drainage (HAND) 
- geomorphic floodplain delineation
- up- and downstream tracing and arithmetics
- hydrologically adjusting elevation
- upstream accumulation
- vectorizing streams
- many more!

.. image:: https://raw.githubusercontent.com/Deltares/pyflwdir/master/docs/_static/pyflwdir.png
  :width: 100%


Installation
------------

See `installation guide <https://deltares.github.io/PyFlwDir/latest/installation.html>`_

Quickstart
----------

See `User guide <https://deltares.github.io/PyFlwDir/latest/quickstart.html>`_


Reference API
-------------

See `reference API <https://deltares.github.io/PyFlwDir/latest/reference.html>`_


Development and Testing
-----------------------

Welcome to the PyFlwDir project. All contributions, bug reports, bug fixes, 
documentation improvements, enhancements, and ideas are welcome. 
See `CONTRIBUTING.rst <CONTRIBUTING.rst/>`__ for how we work.

Changes
-------

See `CHANGELOG.rst <CHANGELOG.rst>`__

Authors
-------

See `AUTHORS.txt <AUTHORS.txt>`__

Citation
--------

For citing our work see the Zenodo badge above, that points to the latest release.

License
-------

This is free software: you can redistribute it and/or modify it under the terms of the
MIT License. A copy of this license is provided in `LICENSE <LICENSE>`__