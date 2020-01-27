import numpy as np
from pyflwdir import core, core_d8, basin, upscale, FlwdirRaster, gis_utils
from pyflwdir.basin_utils import basin_area

_mv = core._mv

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

if __name__ == "__main__":
    from affine import Affine
    from pyflwdir.gis_utils import idxs_to_coords
    from pyflwdir import subgrid
    import rasterio

    cellsize = 10
    method = 'eeam'
    res = 1/1200
    affine = Affine(res, 0.0, -10.5, 0.0, -res, 55.5)
    affine2 = Affine(res*cellsize, 0.0, -10.5, 0.0, -res*cellsize, 55.5)

    d8_data = np.fromfile(r'./tests/data/ireland_d8.bin', dtype=np.uint8).reshape((4920, 6120))
    # parse d8 data
    d8 = FlwdirRaster(d8_data, ftype = 'd8', check_ftype = False)
    uparea = d8.upstream_area(latlon=False).ravel()
    uparea = np.where(uparea!=-9999., uparea/1e2, -9999.)
    basins = d8.basins().ravel()

    # upscale
    # nextidx, subidxs_out = upscale.eam(
    #     d8._idxs_ds, d8._idxs_valid, uparea, d8.shape, cellsize,
    #     return_connect=True)
    nextidx, subidxs_out = upscale.cosm(
        d8._idxs_ds, d8._idxs_valid, uparea, d8.shape, cellsize, 
        iter2='2' in method, return_connect=True)
    dir_lr = FlwdirRaster(nextidx, ftype = 'nextidx', check_ftype = True)
    subidxs_out = subidxs_out.ravel()
    subidx_out_internal = d8._internal_idx(subidxs_out[dir_lr._idxs_valid])
    connect = np.ones(dir_lr.shape, dtype=np.int8)*-1
    connect.flat[dir_lr._idxs_valid] = subgrid.connected(dir_lr._idxs_ds, subidx_out_internal, d8._idxs_ds)
    print(np.sum(connect == 0))

    # check
    assert dir_lr.isvalid
    assert np.unique(nextidx.flat[dir_lr.pits]).size == nextidx.flat[dir_lr.pits].size
    pitbas = basins[subidxs_out[dir_lr.pits]]
    assert np.unique(pitbas).size == pitbas.size
    # check quality
    valid = nextidx.ravel() != _mv
    subbasins = d8.subbasins(subidxs_out[valid])
    subare = np.ones(dir_lr.shape)*-9999.
    # subare.flat[valid] = basin_area(subbasins, affine=affine, latlon=True)
    subare.flat[valid] = basin_area(subbasins, latlon=False)
    uparea_lr = dir_lr.accuflux(subare, nodata=-9999.).ravel()
    uparea_lr = np.where(uparea_lr!=-9999., uparea_lr/1e2, -9999.)
    uparea_out = np.ones(dir_lr.shape).ravel()*-9999.
    uparea_out[valid] = uparea[subidxs_out[valid]]
    nse = _nse(uparea_lr[valid], uparea_out[valid]) 
    upadiff = np.where(uparea_lr!=-9999, (uparea_out - uparea_lr), -9999)
    relbias10 = np.sum(upadiff[valid]/uparea_out[valid]>0.1)/upadiff.size*100
    dupa = np.abs(upadiff[valid])    
    print(f'NSE: {nse:.4f}, BIAS>10%: {relbias10:.4f},  dUPA max: {dupa.max():.4f}, dUPA mean: {dupa.mean():.4f}')

    idxs_error = np.where(np.logical_and(valid, np.abs(upadiff)>500))[0]
    seq = np.argsort(np.abs(upadiff[idxs_error]))
    for i in seq[-min(seq.size, 10):]:
        idx = idxs_error[i]
        print(f'{np.abs(upadiff[idx]):.2f} km2:', idxs_to_coords(idx, affine2, dir_lr.shape)[::-1])
    
    # river length
    import pdb; pdb.set_trace()
    rivlen = np.ones(dir_lr.shape, dtype=np.float64)*-9999.
    rivlen.flat[dir_lr._idxs_valid] = subgrid.river_length(subidx_out_internal, 
        dir_lr._idxs_valid, dir_lr._idxs_ds, dir_lr._idxs_us, 
        d8._idxs_valid, d8._idxs_ds, d8._idxs_us, d8.shape,
        uparea.flat[d8._idxs_valid], min_uparea=1., 
        latlon=False, affine=gis_utils.IDENTITY)

    # write files for visual check
    import pdb; pdb.set_trace()
    valid = nextidx.ravel() != _mv
    xs, ys = np.ones(nextidx.size)*np.nan, np.ones(nextidx.size)*np.nan
    xs[valid], ys[valid] = gis_utils.idxs_to_coords(subidxs_out[valid], affine, d8.shape)
    dir_lr.vector(xs=xs, ys=ys).to_file(f'./tests/data/ireland{cellsize}_{method}.gpkg', layer='rivers', driver="GPKG")
    nodata = -9999.
    data = upadiff.reshape(dir_lr.shape)
    name = f'upadiff_{method}'
    
    # nodata=_mv
    # data = np.ones(nextidx.size, dtype=np.uint32)*_mv
    # data[dir_lr._idxs_valid] = dir_lr._idxs_valid
    # data = data.reshape(dir_lr.shape)
    # name = f'idx'
    # data = d8.subbasins(subidxs_out[valid])
    # name = 'ucat'
    # nodata = 0
    # data = d8.stream_order().astype(np.int32)
    # name = 'ord'
    # nodata = -1
    # transform = affine
    prof = dict(
        driver = 'GTiff',
        height = data.shape[0], 
        width = data.shape[1],
        transform = affine2,
        count = 1,
        dtype = data.dtype,
        nodata = nodata
        )
    with rasterio.open(f'./tests/data/ireland{cellsize}_{name}.tif', 'w', **prof) as dst:
        dst.write(data, 1)

    # idxs_loop = dir_lr._idxs_valid[core.loop_indices(dir_lr._idxs_ds, dir_lr._idxs_us)]
    name = f'connect_{method}'
    nodata = np.int8(-1)
    data = connect
    prof.update(nodata=nodata, dtype=data.dtype)
    with rasterio.open(f'./tests/data/ireland{cellsize}_{name}.tif', 'w', **prof) as dst:
        dst.write(data, 1)

    name = f'basins_{method}'
    nodata = 0
    data = dir_lr.basins()
    prof.update(nodata=nodata, dtype=data.dtype)
    with rasterio.open(f'./tests/data/ireland{cellsize}_{name}.tif', 'w', **prof) as dst:
        dst.write(data, 1)

    name = f'rivlen_{method}'
    nodata = -9999
    data = rivlen
    prof.update(nodata=nodata, dtype=data.dtype)
    with rasterio.open(f'./tests/data/ireland{cellsize}_{name}.tif', 'w', **prof) as dst:
        dst.write(data, 1)