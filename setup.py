#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="pyflwdir",
    description="Fast methods to work with hydro- and topography data in pure Python.",
    long_description=readme + "\n\n",
    url="https://gitlab.com/deltares/wflow/pyflwdir/",
    author="Dirk Eilander",
    author_email="dirk.eilander@deltares.nl",
    license="MIT",
    packages=find_packages(),
    package_dir={"pyflwdir": "pyflwdir"},
    test_suite="tests",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.6",
    install_requires=["numba>=0.48", "numpy", "scipy", "affine",],
    extras_require={
        "dev": ["pytest", "pytest-cov", "sphinx", "sphinx_rtd_theme", "black"],
        "optional": ["geopandas",],
    },
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="hydrology watershed basins stream pyflwdir wflow",
)
