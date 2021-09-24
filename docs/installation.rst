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

In order to run the examples in the notebook folder some aditional packages to read 
and write raster and vector data, as well as to plot these data are required. 
We recommend using `rasterio <https://rasterio.readthedocs.io/>`__ raster data and 
`geopandas <https://geopandas.org/>`__ for vector data.  
A complete environment can be installed from the environment.yml file using:

.. code-block:: console

    $ conda env create -f environment.yml
    $ pip install pyflwdir


Install from github (for developers)
------------------------------------

For we advise the following steps to install the package.

First, clone pyflwdir's ``git`` repo and navigate into the repository:

.. code-block:: console

    $ git clone git@github.com:Deltares/pyflwdir.git
    $ cd pyflwdir

Then, make and activate a new pyflwdir conda environment based on the environment.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate pyflwdir

Finally, build an editable install pyflwdir using pip:

.. code-block:: console

    $ pip install -e .