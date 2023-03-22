Installation
============

Install from conda
------------------

.. code-block:: console

    $ conda install pyflwdir -c conda-forge


Install from pip
----------------

.. code-block:: console

    $ pip install pyflwdir


Install full environment for quickstart and examples
-----------------------------------------------------

In order to run the examples in the examples folder some additional packages to read 
and write raster and vector data, as well as to plot these data are required. 
We recommend using `rasterio <https://rasterio.readthedocs.io/>`__ raster data and 
`geopandas <https://geopandas.org/>`__ for vector data.  
A complete environment can be installed from the environment.yml file using:

.. code-block:: console

    $ conda env create -f environment.yml