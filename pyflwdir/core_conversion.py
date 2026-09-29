"""Methods to convert between different flwdir types"""

import numpy as np

from . import core_d8, core_ldd

__all__ = ["d8_to_ldd", "ldd_to_d8"]


def d8_to_ldd(flwdir: np.ndarray) -> np.ndarray:
    """Convert an ArcGIS D8 flow-direction array to PCRaster LDD codes.

    Parameters
    ----------
    flwdir : 2D array of int
        Flow-direction values using the ArcGIS D8 convention.

    Returns
    -------
    2D array of int
        Flow directions using the PCRaster LDD convention. Values not recognized as
        valid D8 codes are converted to the LDD no-data value.
    """
    # create conversion dict
    remap: dict = {
        k: v for (k, v) in zip(core_d8._ds.flatten(), core_ldd._ds.flatten())
    }
    # add addional land pit code to pcr pit
    remap.update({core_d8._pv[1]: core_ldd._pv, core_d8._mv: core_ldd._mv})
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_ldd._mv))(flwdir)


def ldd_to_d8(flwdir: np.ndarray) -> np.ndarray:
    """Convert a PCRaster LDD flow-direction array to ArcGIS D8 codes.

    Parameters
    ----------
    flwdir : 2D array of int
        Flow-direction values using the PCRaster LDD convention.

    Returns
    -------
    2D array of int
        Flow directions using the ArcGIS D8 convention. Values not recognized as valid
        LDD codes are converted to the D8 no-data value.
    """
    # create conversion dict
    remap: dict = {
        k: v for (k, v) in zip(core_ldd._ds.flatten(), core_d8._ds.flatten())
    }
    # add addional land pit code to pcr pit
    remap.update({core_ldd._pv: core_d8._pv[0], core_ldd._mv: core_d8._mv})
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_d8._mv))(flwdir)
