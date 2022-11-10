import argparse
import configparser
import os
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal, osr


def rasterize(fin_shape:str, fin_raster:str, dir_out:str, field_target:str, field_id:str, target_to_int:pd.DataFrame, nanval:int=-9999):
    """Converts label (vector poligon) to raster

    Args:
        fin_shape (str): input vector poligon to be rasterized
        fin_raster (str): template raster
        dir_out (str): output directory
        field_target (str): attribute name to be used for rasterizing
        field_id (str): attribute with object identifier (needs to be an int)
        target_to_int (pd.DataFrame): a pd Dataframe that relates what integer to be used for each target value
            (rasters values cannot be strings)
        nanval (int, optional): value to be used as No Data/NaN in the rasterized output. Defaults to -9999.
    """

    # check if output directory exists, creats if it doesn't
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    # set up raster names
    fout_temp = os.path.join(dir_out, 'temp.tif')
    fout = os.path.join(dir_out,
                        f'{Path(fin_shape).resolve().stem}-{field_target}.tif')

    # use the provided raster as template
    template_raster = gdal.Open(fin_raster)

    # setup a new raster
    drv_tiff = gdal.GetDriverByName("GTiff")
    
    proj = osr.SpatialReference()
    proj.ImportFromWkt(template_raster.GetProjectionRef())

    rasterized = drv_tiff.Create(fout_temp,
                                    template_raster.RasterXSize, template_raster.RasterYSize,
                                    1,
                                    gdal.GDT_Int16)
    rasterized.SetGeoTransform(template_raster.GetGeoTransform())
    rasterized.SetProjection(proj.ExportToWkt())

    # close template raster
    template_raster = None

    # set the "No Data Value"
    rasterized_band = rasterized.GetRasterBand(1)
    rasterized_band.Fill(nanval)
    rasterized.GetRasterBand(1).SetNoDataValue(nanval)

    # open the shape to be rasterized
    shape_in = gdal.OpenEx(fin_shape)
    # get the layer shape
    lyr = shape_in.GetLayer()

    # create a dictionary that maps field_id to field_target
    id_target_dict = {}

    for feature in lyr:
        id_target_dict[feature.GetField(
            field_id)] = feature.GetField(field_target)


    # rasterize the shape
    # needs numeric attribute!
    gdal.RasterizeLayer(rasterized, [1], lyr,
                        options=["ALL_TOUCHED=TRUE",
                                    f"ATTRIBUTE={field_id}"])

    # close to write the raster
    rasterized = None

    # close the input shapefile
    shape_in = None

    # create a new raster
    gdal.Translate(fout, fout_temp)

    # reopen raster to replace field_id by field_target:
    rasterized = gdal.Open(fout, 1)
    band = rasterized.GetRasterBand(1)
    band_np = band.ReadAsArray()

    # replace field_id by field_target
    out = np.empty(band_np.shape, dtype='U25')
    for key, val in id_target_dict.items():
        idx = band_np == int(key)
        out[idx] = str(val)

    # label encoding for the project:  
    lab_to_int = dict(zip(target_to_int[field_target].astype(str),
                                target_to_int[field_id].astype(int)))

    out = pd.Series(out.ravel()).map(lab_to_int).to_numpy().reshape(band_np.shape)
    out[band.GetMaskBand().ReadAsArray() == 0] = nanval
    band.WriteArray(out)

    # close to write
    rasterized = None

    # delete temporary raster:
    os.remove(fout_temp)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='utils/config.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        field_target = [val.strip() for val in config['dict_class_int']['field_target'].split(',')]
        field_id = [int(val) for val in config['dict_class_int']['field_id'].split(',')]

        df = pd.DataFrame(list(zip(field_target, field_id)), 
                          columns=[config['params']['field_target'], config['params']['field_id']])

        fin_shape_list = [val for val in config['io']['fin_shape_list'].split('\n')]
        fin_raster_list = [val for val in config['io']['fin_raster_list'].split('\n')]


        for fin_shape, fin_raster in zip(fin_shape_list, fin_raster_list):

            print(f'Processing {fin_shape}, {fin_raster}')
            
            rasterize(fin_shape=fin_shape, 
                    fin_raster=fin_raster, 
                    dir_out=config['io']['dir_out'], 
                    field_target=config['params']['field_target'], 
                    field_id=config['params']['field_id'], 
                    target_to_int=df, 
                    nanval=-9999)
