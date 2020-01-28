# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

import logging

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Dirk Eilander"
__email__ = 'dirk.eilander@deltares.nl'

from pyflwdir.pyflwdir import *
from pyflwdir.slope import *