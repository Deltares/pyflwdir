"""Fast methods to work with hydro- and topography data in pure Python."""

# version number without 'v' at start
__version__ = "0.6.0"

# submodules
from . import gis_utils, partition, regions

# public functions
from .core_conversion import d8_to_ldd, ldd_to_d8
from .core_nextxy import read_nextxy
from .dem import fill_depressions, slope
from .flwdir import Flwdir, from_dataframe
from .pyflwdir import FlwdirRaster, from_array, from_dem

__all__ = [
    "Flwdir",
    "FlwdirRaster",
    "from_array",
    "from_dataframe",
    "from_dem",
    "read_nextxy",
    "gis_utils",
    "partition",
    "regions",
    "slope",
    "fill_depressions",
    "d8_to_ldd",
    "ldd_to_d8",
]
