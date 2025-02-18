"""Fast methods to work with hydro- and topography data in pure Python."""

# version number without 'v' at start
__version__ = "0.5.10"

# submodules
from . import gis_utils, regions

# public functions
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
    "regions",
    "slope",
    "fill_depressions",
]
