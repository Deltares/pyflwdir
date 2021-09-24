import pkg_resources

# submodules
from .flwdir import *
from .pyflwdir import *
from .dem import *

from . import gis_utils, regions

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
