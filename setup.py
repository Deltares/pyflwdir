#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit pyflwdir/__version__.py
version = {}
with open(os.path.join(here, 'pyflwdir', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='pyflwdir',
    version=version['__version__'],
    description="Fast numba-based library with flow direction and watershed delineation operators in pure Python",
    long_description=readme + '\n\n',
    author="Dirk Eilander",
    author_email='dirk.eilander@deltares.nl',
    url='https://gitlab.com/deltares/wflow/pyflwdir/',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    license="GNU General Public License v3 or later",
    zip_safe=False,
    keywords='pyflwdir',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy',
        'numba',
        'rasterio',
        'xarray',
        'geopandas'
    ],  
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
