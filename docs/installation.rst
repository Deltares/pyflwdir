Installation
============

Install from pip
----------------

.. code-block:: console

    $ conda install pyflwdir -c conda-forge


Install from pip
----------------

.. code-block:: console

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