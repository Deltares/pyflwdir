# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

import glob
from osgeo import gdal
from os.path import join, isfile, basename, dirname
import os
import xarray as xr
from rasterio.transform import Affine, array_bounds, from_bounds
import numpy as np
import rasterio
import dask
from dask.diagnostics import ProgressBar


def create_empty_vrt(fn_vrt, nodata, bbox, res, dtype):
    w,s,e,n = bbox
    width, height = int(np.round(abs(e-w)/res,0)), int(np.round(abs(n-s)/res,0))
    dtype = rasterio.dtypes._gdal_typename(dtype)
    vrt = f"""<VRTDataset rasterXSize="{width:d}" rasterYSize="{height:d}">
  <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]</SRS>
  <GeoTransform> {w:+.16e},  {res:.16e},  {0.:.16e},  {n:+.16e},  {0:.16e}, {-res:+.16e}</GeoTransform>
  <VRTRasterBand dataType="{dtype:s}" band="1">
    <NoDataValue>{nodata:.0f}</NoDataValue>
    <ColorInterp>Gray</ColorInterp>
  </VRTRasterBand>
</VRTDataset>"""
    with open(fn_vrt, 'w') as f:
        f.writelines(vrt)

def build_vrt(fn_vrt, fns, bbox=None, nodata=None):
    vrt_options = gdal.BuildVRTOptions(
        outputBounds=bbox, 
        srcNodata=nodata, 
        VRTNodata=nodata
    )
    if isfile(fn_vrt):
        os.remove(fn_vrt)
    vrt = gdal.BuildVRT(fn_vrt, fns, options=vrt_options)
    if vrt is None:
        raise Exception('Creating vrt not successfull, check input files.')
    vrt = None # to write to disk

def open_vrt(fn_vrt, size):
    da = xr.open_rasterio(fn_vrt, chunks={'y':size[0], 'x':size[1]})
    da = da.rename({'x':'lon', 'y':'lat'}).drop('band').squeeze()
    transform = Affine(*da.attrs['transform'])
    height, width = da.shape
    bbox = np.array(array_bounds(height, width, transform)).round(8)
    da.attrs.update(bbox=bbox)
    return da

def open_vrt_dataset(root, names, size=(6000, 6000)):
    dataset = []
    fns = []
    for name in names:
        fn_vrt = join(root, f'{name}.vrt')
        if not isfile(fn_vrt):
            raise IOError(f'{fn_vrt} not found')
        fns.append(basename(fn_vrt))
        da = open_vrt(fn_vrt, size)
        da.name = name
        dataset.append(da)
    ds = xr.merge(dataset)
    ds.attrs.update(source_files='; '.join(fns))
    return ds

def create_empty_tiles_like(fns, name, bbox=(-180.,  -60.,  180.,   85.), nodata=-9999, dtype=np.int32, **kwargs):
    @dask.delayed
    def _empty_tile(fn, name, nodata, dtype, **kwargs):
        with rasterio.open(fn) as src:
            prof = src.profile
            if 'compress' in prof:
                prof.pop('compress')
            prof.update(dtype=dtype, nodata=nodata)
            prof.update(**kwargs)
            transform, height, width = prof['transform'], prof['height'], prof['width']
            w, s, e, n = np.asarray(array_bounds(height, width, transform)).round(0)
            n_s = 's' if s < 0 else 'n'
            e_w = 'w' if w < 0 else 'e'
            tile_name = f'{n_s:s}{abs(s):02.0f}{e_w:s}{abs(w):03.0f}'
            fn_out = join(dirname(fn), f'{tile_name}_{name}.tif')
            data = np.ones((height, width), dtype=dtype)*nodata
            with rasterio.open(fn_out, 'w', **prof) as dst:
                dst.write(data, 1)
    
    tasks = []
    for fn in fns:
        tasks.append(_empty_tile(fn, name, nodata, dtype, **kwargs))
    with ProgressBar():    
        dask.compute(*tasks, scheduler='threads')
    
    root = dirname(fns[0])
    fns_out = glob.glob(join(root, f'*{name}.tif'))
    fn_vrt = join(root, f'{name}.vrt')
    build_vrt(fn_vrt, fns_out, bbox=bbox, nodata=nodata)

def append_gtiff_tiles(da, bbox, root, postfix, tile_res, tile_shape, glob_bbox, force_overwrite=False, **kwargs):
    w, s, e, n = bbox
    xs = np.arange(w, e, tile_res)
    ys = np.arange(n, s, -tile_res) # assume north > south
    tile_height, tile_width = tile_shape
    for w in xs:
        e = w + tile_res
        sel = dict(lon=slice(w, e)) # NOTE based on adjusted axis
        if np.round(e,8) > glob_bbox[2]: # glob bbox is saved with precision 8
            e -= 360
            w -= 360
        elif np.round(w,8) < glob_bbox[0]:
            e += 360
            w += 360
        for n in ys:
            s = n - tile_res
            sel.update(lat=slice(n, s))
            tile_transform = from_bounds(w, s, e, n, tile_width, tile_height) 
            kwargs.update(transform=tile_transform)
            # name
            n_s = 's' if s < 0 else 'n'
            e_w = 'w' if w < 0 else 'e'
            tile_name = f'{n_s:s}{abs(s):02.0f}{e_w:s}{abs(w):03.0f}'
            # if 'e180' in tile_name or 'w185' in tile_name:
            #     import pdb; pdb.set_trace()
            fn = join(root, f'{tile_name}_{postfix}.tif')
            # get data
            tile_data = da.sel(**sel).values
            assert np.all(tile_data.shape == tile_shape)
            if np.all(tile_data==-9999): continue
            if isfile(fn):
                with rasterio.open(fn, 'r') as src:
                    tile_data0 = src.read(1)
                    if not force_overwrite:
                        assert np.all(np.logical_or(
                        tile_data0[tile_data!=-9999] == -9999,
                        tile_data0[tile_data!=-9999] == tile_data[tile_data!=-9999]
                        ))
                    tile_data = np.where(tile_data==-9999, tile_data0, tile_data)
            with rasterio.open(fn, 'w', **kwargs) as dst:
                dst.write(tile_data, 1)