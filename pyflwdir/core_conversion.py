# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

"""Convert between different flwdir types"""

import numpy as np
from pyflwdir import core_d8, core_ldd

def d8_to_ldd(flwdir):
    # create conversion dict
    remap = {k:v for (k,v) in zip(core_d8._ds.flatten(), core_ldd._ds.flatten())}
    # add addional land pit code to pcr pit
    remap.update({
        core_d8._pv[1]: core_ldd._pv, 
        core_d8._mv: core_ldd._mv
        })
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_ldd._mv))(flwdir)

def ldd_to_d8(flwdir):
    # create conversion dict
    remap = {k:v for (k,v) in zip(core_ldd._ds.flatten(), core_d8._ds.flatten())}
    # add addional land pit code to pcr pit
    remap.update({
        core_ldd._pv: core_d8._pv[0], 
        core_ldd._mv: core_d8._mv
        })
    # remap values
    return np.vectorize(lambda x: remap.get(x, core_d8._mv))(flwdir)