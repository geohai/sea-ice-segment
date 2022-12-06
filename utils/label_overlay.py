import os
from pathlib import Path

from osgeo import gdal

def label_overlay(fin_label_raster:str, fin_input_raster:str):
    """Overlays a label raster on top of an input raster. 
    Marks pixels in the label raster as no data if the input is nan

    Args:
        fin_label_raster (str): label raster to be cleaned
        fin_input_raster (str): input raster that will be used to predict label
    """

    # open the label and input rasters:
    label_raster = gdal.Open(fin_label_raster, 1)
    input_raster = gdal.Open(fin_input_raster)

    # get the "No Data" Mask
    band_input = input_raster.GetRasterBand(1)

    # modify the label raster
    band_label = label_raster.GetRasterBand(1)
    out = band_label.ReadAsArray()
    
    nanval = band_label.GetNoDataValue()
    out[band_input.GetMaskBand().ReadAsArray() == 0] = nanval
    band_label.WriteArray(out)

    # close to write
    label_raster = None

    # close input raster
    input_raster = None

if __name__ == '__main__':

    fin_raster_list = [
                        'seaice_s1_20180116t075430-NE-SA.tif',
                        'seaice_s1_20180612t180423-NE-SA.tif',
                        'seaice_s1_20180717t073809-NE-SA.tif',
                        'seaice_s1_20181218t075437-NE-SA.tif',
                        'seaice_s1_20180116t075430-SW-SA.tif',
                        'seaice_s1_20180612t180423-SW-SA.tif',
                        'seaice_s1_20180717t073809-SW-SA.tif',
                        'seaice_s1_20181218t075437-SW-SA.tif',
                      ]

    fin_label_raster =['20180116t075430-NE.tif',
                      '20180612t180423-NE.tif',
                      '20180717t073809-NE.tif',
                      '20181218t075437-NE.tif',
                      '20180116t075430-SW.tif',
                      '20180612t180423-SW.tif',
                      '20180717t073809-SW.tif',
                      '20181218t075437-SW.tif',
                     ]
    
    dir_in_raster = 'E:/rafael/data/Extreme_Earth/labels_rasterized/SA'
    dir_in_label = 'E:/rafael/data/Extreme_Earth/denoised_resampled'

    for f_label, f_raster in zip(fin_label_raster, fin_raster_list):

        fin_label = os.path.join(dir_in_label, f_label)
        fin_raster = os.path.join(dir_in_raster, f_raster)

        print(f'Processing {fin_label}, {fin_raster}')
        
        label_overlay(fin_label, fin_raster)

    
