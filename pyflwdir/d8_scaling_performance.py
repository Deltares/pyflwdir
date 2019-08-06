# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import jit
import numpy as np
from os.path import join
import xarray as xr

# set d8 format
from d8_func import d8
_d8_pit = d8._d8[1,1]

# correlation ufunc function
def _covariance(x, y):
    return np.nanmean((x - np.nanmean(x, axis=-1, keepdims=True))
            * (y - np.nanmean(y, axis=-1, keepdims=True)), axis=-1)
def _pearson_correlation(x, y):
    return _covariance(x, y) / (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))
def _rsquared(x, y):
    return _pearson_correlation(x, y) ** 2
def _nse(sim, obs, axis=-1):
    """nash-sutcliffe efficiency"""
    obs_mean = np.nanmean(obs, axis=axis)
    a = np.nansum((sim-obs)**2, axis=axis)
    b = np.nansum((obs-obs_mean[..., None])**2, axis=axis)
    return 1 - a/b

# TODO add CSI 

def test_scaling(upa_scaled, upa_outlet):
    valid = np.logical_and(upa_outlet!=-9999, upa_scaled!=-9999)
    # r2 = _rsquared(upa_scaled[valid], upa_outlet[valid], )
    # return r2
    me = _nse(upa_scaled[valid], upa_outlet[valid], )
    # if me > 1.:
    #     import pdb; pdb.set_trace()
    return me

def scatterplot(fn, uparea, outupa, txt=''):
    # romove lower values for plot
    outupa = outupa[uparea>1]
    uparea = uparea[uparea>1]
    # figure
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(1,1, figsize=(8,8))
    ax1.set_aspect('equal')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.scatter(x=uparea, y=outupa, color='k', s=.1)
    # ax1.scatter(x, y, c=z, s=1, edgecolor='')
    ax1.set_xlabel('upstream area at outlet [km2]')
    ax1.set_ylabel('scaled upstream area [km2]')
    if txt != '':
        plt.text(0.95, 0.05, txt, 
            horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes,
            bbox=dict(edgecolor='k', facecolor='w', alpha=1))
    plt.savefig(fn, dpi=255, bbox_axis='tight')
    plt.close()
    return