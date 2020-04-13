import pkg_resources

# submodules
from pyflwdir.pyflwdir import *
from pyflwdir import dem, gis_utils

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
