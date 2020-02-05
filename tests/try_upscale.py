import numpy as np
from os.path import join
from pyflwdir import core, core_d8, basin, upscale, FlwdirRaster, gis_utils
from pyflwdir.basin_utils import basin_area
from affine import Affine
from pyflwdir.gis_utils import idxs_to_coords
from pyflwdir import subgrid

_mv = core._mv


def _covariance(x, y):
    return np.nanmean((x - np.nanmean(x, axis=-1, keepdims=True)) *
                      (y - np.nanmean(y, axis=-1, keepdims=True)),
                      axis=-1)


def _pearson_correlation(x, y):
    return _covariance(x, y) / (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))


def _rsquared(x, y):
    return _pearson_correlation(x, y)**2


def _nse(sim, obs, axis=-1):
    """nash-sutcliffe efficiency"""
    obs_mean = np.nanmean(obs, axis=axis)
    a = np.nansum((sim - obs)**2, axis=axis)
    b = np.nansum((obs - obs_mean[..., None])**2, axis=axis)
    return 1 - a / b


if __name__ == "__main__":
    import rasterio  # NOTE not a dependecy

    cellsize = 10
    method = 'eeam2'

    # prefix = 'rhine'
    # d8_data = np.fromfile(r'./tests/data/d8.bin', dtype=np.uint8)
    # d8_data = d8_data.reshape((678, 776))
    # res, west, north = 1 / 120, 5 + 50 / 120., 51 + 117 / 120.
    # affine = Affine(res, 0.0, west, 0.0, -res, north)

    prefix = 'n55w015' #'ireland' #'n30w100'
    with rasterio.open(f'./tests/data/{prefix}_dir.tif', 'r') as src:
        d8_data = src.read(1)
        affine = src.transform
    res, west, north = affine[0], affine[2], affine[5]
    with rasterio.open(f'./tests/data/{prefix}_upa.tif', 'r') as src:
        uparea = src.read(1).ravel()

    affine2 = Affine(res * cellsize, 0.0, west, 0.0, -res * cellsize, north)
    import pdb; pdb.set_trace()
    # parse d8 data
    d8 = FlwdirRaster(d8_data, ftype='d8', check_ftype=False)
    # uparea = d8.upstream_area(latlon=False).ravel()
    # uparea = np.where(uparea != -9999., uparea / 1e2, -9999.)

    prof = dict(driver='GTiff',
                height=d8.shape[0],
                width=d8.shape[1],
                transform=affine,
                count=1)
    # celledge = upscale.map_celledge(d8._idxs_ds, d8._idxs_valid, d8.shape, cellsize).astype(np.uint8)
    # effare = upscale.map_effare(d8._idxs_ds, d8._idxs_valid, d8.shape, cellsize).astype(np.uint8)
    # nodata = 255
    # data = d8.stream_order().astype(np.int32)
    # name = 'ord'
    # nodata = -1
    # name = 'upa'
    # nodata = -9999.
    # data = uparea.reshape(d8.shape)
    # prof.update(nodata=nodata, dtype=data.dtype)
    # fn = f'./tests/data/{prefix}0_{name}.tif'
    # with rasterio.open(fn, 'w', **prof) as dst:
    #     dst.write(data, 1)
    # import pdb; pdb.set_trace()

    # upscale
    # nextidx, subidxs_out = upscale.eam(
    #     d8._idxs_ds, d8._idxs_valid, uparea, d8.shape, cellsize)
    nextidx, subidxs_out = upscale.cosm(d8._idxs_ds,
                                        d8._idxs_valid,
                                        uparea,
                                        d8.shape,
                                        cellsize,
                                        iter2='2' in method)
    # shape = (610, 610)
    # fn_nextidx = f'./tests/data/{prefix}{cellsize}_nextidx.bin'
    # nextidx = np.fromfile(fn_nextidx, dtype=np.uint32)
    # fn_subidxs_out = f'./tests/data/{prefix}{cellsize}_subidxs_out.bin'
    # subidxs_out = np.fromfile(fn_subidxs_out, dtype=np.uint32)
    # idxs_fix = np.array([197917], dtype=np.uint32)
    # nextidx, subidxs_out = upscale.cosm_nextidx_iter2(nextidx, subidxs_out,
    #                                         idxs_fix, d8._idxs_ds,
    #                                         d8._idxs_valid, uparea,
    #                                         d8.shape, shape, cellsize)
    # nextidx, subidxs_out = nextidx.reshape(shape), subidxs_out.reshape(shape)
    dir_lr = FlwdirRaster(nextidx, ftype='nextidx', check_ftype=True)

    prof2 = dict(driver='GTiff',
                height=dir_lr.shape[0],
                width=dir_lr.shape[1],
                transform=affine2,
                count=1)

    repair_idx = core.loop_indices(dir_lr._idxs_ds, dir_lr._idxs_us)
    if repair_idx.size > 0:
        repair_idx = dir_lr._idxs_valid[repair_idx]
        data = np.zeros(dir_lr.shape, dtype=np.int8)
        data.flat[repair_idx] = 1
        name = f'loop_{method}'
        prof2.update(nodata=0, dtype=data.dtype)
        import pdb; pdb.set_trace()
        fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
        with rasterio.open(fn, 'w', **prof2) as dst:
            dst.write(data, 1)
        fn = f'./tests/data/{prefix}{cellsize}_nextidx.bin'
        nextidx.tofile(fn)
        fn = f'./tests/data/{prefix}{cellsize}_subidxs_out.bin'
        subidxs_out.tofile(fn)
        print(idxs_to_coords(repair_idx[0], affine2, dir_lr.shape)[::-1])
        print(repair_idx.size)

    dir_lr.to_array(ftype='d8')
    subidxs_out = subidxs_out.ravel()
    subidxs_out0 = d8._internal_idx(subidxs_out[dir_lr._idxs_valid])
    connect = np.ones(dir_lr.shape, dtype=np.int8) * -1
    connect.flat[dir_lr._idxs_valid] = subgrid.connected(
        subidxs_out0, dir_lr._idxs_ds, d8._idxs_ds)
    print(np.sum(connect == 0))
    # print(idxs_to_coords(165, affine2, dir_lr.shape)[::-1])

    # check
    assert np.unique(
        nextidx.flat[dir_lr.pits]).size == nextidx.flat[dir_lr.pits].size
    basins = d8.basins().ravel()
    pitbas = basins[subidxs_out[dir_lr.pits]]
    assert np.unique(pitbas).size == pitbas.size
    # check quality
    valid = nextidx.ravel() != _mv
    subare = np.ones(dir_lr.shape, dtype=np.float32) * -9999.
    subare.flat[dir_lr._idxs_valid] = subgrid.cell_area(
        subidxs_out0, d8._idxs_valid, d8._idxs_us, d8.shape).astype(np.float32)
    uparea_lr = dir_lr.accuflux(subare, nodata=-9999.).ravel()
    uparea_lr = np.where(uparea_lr != -9999., uparea_lr / 1e2, -9999.)
    uparea_out = np.ones(dir_lr.shape).ravel() * -9999.
    uparea_out[valid] = uparea[subidxs_out[valid]]
    nse = _nse(uparea_lr[valid], uparea_out[valid])
    upadiff = np.where(uparea_lr != -9999, (uparea_out - uparea_lr), -9999)
    relbias10 = np.sum(
        upadiff[valid] / uparea_out[valid] > 0.1) / upadiff.size * 100
    dupa = np.abs(upadiff[valid])
    print(f'NSE: {nse:.4f}, BIAS>10%: {relbias10:.4f}, ' +
          f'dUPA max: {dupa.max():.4f}, dUPA mean: {dupa.mean():.4f}')

    idxs_error = np.where(np.logical_and(valid, np.abs(upadiff) > 500))[0]
    seq = np.argsort(np.abs(upadiff[idxs_error]))
    for i in seq[-min(seq.size, 10):]:
        idx = idxs_error[i]
        print(f'{np.abs(upadiff[idx]):.2f} km2:',
              idxs_to_coords(idx, affine2, dir_lr.shape)[::-1])
    import pdb; pdb.set_trace()

    # river length
    rivlen = np.ones(dir_lr.shape, dtype=np.float32) * -9999.
    rivlen.flat[dir_lr._idxs_valid] = subgrid.river_params(
        subidxs_out0, #[dir_lr._internal_idx(np.array([1440]))],
        d8._idxs_valid,
        d8._idxs_ds,
        d8._idxs_us,
        subelevtn=np.ones(d8.ncells),  # fake elevation; ignore slope
        subuparea=uparea.flat[d8._idxs_valid],
        subshape=d8.shape,
        min_uparea=0.,
        latlon=False,
        affine=gis_utils.IDENTITY)[0].astype(np.float32)

    # write files for visual check
    valid = nextidx.ravel() != _mv
    xs, ys = np.ones(nextidx.size) * np.nan, np.ones(nextidx.size) * np.nan
    xs[valid], ys[valid] = gis_utils.idxs_to_coords(subidxs_out[valid], affine, d8.shape)
    dir_lr.vector(xs=xs, ys=ys).to_file(
        f'./tests/data/{prefix}{cellsize}_{method}.gpkg',
        layer='rivers',
        driver="GPKG")
    
    # highres
    # name = 'ucat'
    # nodata = 0
    # data = d8.subbasins(subidxs_out[valid])
    # prof.update(nodata=nodata, dtype=data.dtype)
    # fn = f'./tests/data/{prefix}0_{name}.tif'
    # with rasterio.open(fn, 'w', **prof) as dst:
    #     dst.write(data, 1)
    
    # lowres
    nodata = -9999.
    data = upadiff.reshape(dir_lr.shape)
    name = f'upadiff_{method}'
    prof2.update(nodata=nodata, dtype=data.dtype)
    fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
    with rasterio.open(fn, 'w', **prof2) as dst:
        dst.write(data, 1)

    # idxs_loop = dir_lr._idxs_valid[core.loop_indices(dir_lr._idxs_ds, dir_lr._idxs_us)]
    name = f'connect_{method}'
    nodata = np.int8(-1)
    data = connect
    prof2.update(nodata=nodata, dtype=data.dtype)
    fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
    with rasterio.open(fn, 'w', **prof2) as dst:
        dst.write(data, 1)

    # name = f'basins_{method}'
    # nodata = 0
    # data = dir_lr.basins()
    # prof2.update(nodata=nodata, dtype=data.dtype)
    # fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
    # with rasterio.open(fn, 'w', **prof2) as dst:
    #     dst.write(data, 1)

    # name = f'rivlen_{method}'
    # nodata = -9999
    # data = rivlen
    # prof2.update(nodata=nodata, dtype=data.dtype)
    # fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
    # with rasterio.open(fn, 'w', **prof2) as dst:
    #     dst.write(data, 1)

    # name = f'subare_{method}'
    # nodata = -9999
    # data = subare
    # prof2.update(nodata=nodata, dtype=data.dtype)
    # fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
    # with rasterio.open(fn, 'w', **prof2) as dst:
    #     dst.write(data, 1)

    name = f'idx'
    nodata = _mv
    data = np.ones(nextidx.size, dtype=np.uint32) * _mv
    data[dir_lr._idxs_valid] = dir_lr._idxs_valid
    data = data.reshape(dir_lr.shape)
    prof2.update(nodata=nodata, dtype=data.dtype)
    fn = f'./tests/data/{prefix}{cellsize}_{name}.tif'
    with rasterio.open(fn, 'w', **prof2) as dst:
        dst.write(data, 1)


