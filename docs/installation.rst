Installation
============

We are working on making the package available through pip and conda package managers.
For we advise the following steps to install the package.

First, clone pyflwdir's ``git`` repo and navigate into the repository:

.. code-block:: console

    $ git clone git@gitlab.com:deltares/wflow/pyflwdir.git
    $ cd pyflwdir

Then, make and activate a new pyflwdir conda environment based on the environment.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate pyflwdir

Finally, build and install pyflwdir using pip:

.. code-block:: console

    $ pip install .