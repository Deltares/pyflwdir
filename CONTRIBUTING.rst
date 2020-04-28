Welcome to the pyflwdir project. All contributions, bug reports, bug fixes, 
documentation improvements, enhancements, and ideas are welcome. Here's how we work.

Code of Conduct
---------------

First of all: the pyflwdir project has a code of conduct. Please read the
CODE_OF_CONDUCT.txt file, it's important to all of us.

Rights
------

The MIT license (see LICENSE.txt) applies to all contributions.

Issue Conventions
-----------------

The pyflwdir issue tracker is for actionable issues.

Pyflwdir is a relatively new project and highly active. We have bugs, both
known and unknown.

Please search existing issues, open and closed, before creating a new one.

Please provide these details as well as tracebacks and relevant logs. Short scripts and 
datasets demonstrating the issue are especially helpful!

Design Principles
-----------------

Pyflwdir contains methods to work with hydro- and topography data in Numpy arrays. 

- I/O from the filesystem to numpy arrays is not part of this package, other packages
  such as `xarray <https://github.com/pydata/xarray>`__. or 
  `rasterio <https://github.com/pydata/xarray>`__. 
- To accalerate the code we use `numba <https://github.com/numba/numba>`__. 
- Flow direction data is parsed to a flattened array of linear indices of the next 
  downstream cell. Based on this general concept many methods can be applied.


Git Conventions
---------------

First of all, if git is new to you, here are some great resources for learning Git:

- the GitLab `help pages <https://gitlab.com/help/gitlab-basics/start-using-git.md>`__.
- the NumPy’s `documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`__.

The code is hosted on GitLab. To contribute you will need to sign up for a free 
GitLab account. We follow the `GitLab workflow 
<https://about.gitlab.com/blog/2016/10/25/gitlab-workflow-an-overview/>`__
to allow many people to work together on the project.

After discussing a new proposal or implementation in the issue tracker, you can start 
working on the code. You write your code locally in a new branch pyflwdir repo or in a 
branch of a fork. Once you're done with your first iteration, you commit your code and 
push to your pyflwdir repository. 

To create a new branch after you've downloaded the latest changes in the project: 

.. code-block:: console

    $ git pull 
    $ git checkout -b <name-of-branch>

Develop your new code and keep while keeping track of the status and differences using:

.. code-block:: console

    $ git status 
    $ git diff

Add and commit local changes, use clear commit messages and add the number of the 
related issue to that (first) commit message:

.. code-block:: console

    $ git add <file-name OR folder-name>
    $ git commit -m "this is my commit message. Ref #xxx"

Regularly push local commits to the repository. For a new branch the remote and name 
of branch need to be added.

.. code-block:: console

    $ git push <remote> <name-of-branch> 

When your changes are ready for review, you can merge them into the main codebase with a 
merge request. We recommend creating a merge request as early as possible to give other 
developers a heads up and to provide an opportunity for valuable early feedback. You 
can create a merge request online or by pushing your branch to a feature-branch. 

Code Conventions
----------------

We use `black <https://black.readthedocs.io/en/stable/>`__ for standardized code formatting.

Tests are mandatory for new features. We use `pytest <https://pytest.org>`__. All tests
should go in the tests directory.

During Continuous Integration testing, several tools will be run to check your code for 
based on pytest, but also stylistic errors.

Development Environment
-----------------------

Developing pyflwdir requires Python >= 3.6. We prefer developing with the most recent 
version of Python. We strongly encourage you to develop in a seperate conda environment.
All Python dependencies required to develop pyflwdir can be found in `environment.yml <environment.yml>`__.

Initial Setup
^^^^^^^^^^^^^

First, clone pyflwdir's ``git`` repo and navigate into the repository:

.. code-block:: console

    $ git clone git@gitlab.com:deltares/wflow/pyflwdir.git
    $ cd pyflwdir

Then, make and activate a new pyflwdir conda environment based on the environment.yml 
file contained in the repository:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate pyflwdir

Finally, build and install pyflwdir:

.. code-block:: console

    $ pip install -e .

Running the tests
^^^^^^^^^^^^^^^^^

Pyflwdir's tests live in the tests folder and generally match the main package layout. 
Test should be run from the tests folder.

To run the entire suite and the code coverage report:

.. code-block:: console

    $ cd tests
    $ python -m pytest --verbose --cov=pyflwdir --cov-report term-missing

A single test file:

.. code-block:: console

    $ python -m pytest --verbose test_pyflwdir.py

A single test:

.. code-block:: console

    $ python -m pytest --verbose test_pyflwdir.py::test_save

Running code format checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

The code formatting will be checked based on the `black clode style 
<https://black.readthedocs.io/en/stable/the_black_code_style.html>`__ during ci. 
Make sure the check below returns *All done!* before commiting your edits.

To check the formatting of your code:

.. code-block:: console

    $ black --check . 

To automatically reformat your code:

.. code-block:: console

    $ black . 