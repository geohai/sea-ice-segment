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

    # fin_label_raster = ['NIS_arctic_20220404_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220405_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220406_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220407_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220408_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220411_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220412_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220413_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220419_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220420_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220503_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220504_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220505_pl_a_sarmap2-NIS_CLASS.tif',
    #                   'NIS_arctic_20220506_pl_a_sarmap2-NIS_CLASS.tif',
    #                   ]

    # fin_raster_list =['S1_merged_20220404.tif',
    #                   'S1_merged_20220405.tif',
    #                   'S1_merged_20220406.tif',
    #                   'S1_merged_20220407.tif',
    #                   'S1_merged_20220408.tif',
    #                   'S1_merged_20220411.tif',
    #                   'S1_merged_20220412.tif',
    #                   'S1_merged_20220413.tif',
    #                   'S1_merged_20220419.tif',
    #                   'S1_merged_20220420.tif',
    #                   'S1_merged_20220503.tif',
    #                   'S1_merged_20220504.tif',
    #                   'S1_merged_20220505.tif',
    #                   'S1_merged_20220506.tif',
    #                  ]
    
    # dir_in_label = 'E:/rafael/data/MET_sarmap/labels_rasterized/NIS_CLASS'
    # dir_in_raster = 'E:/rafael/data/MET_sarmap/S1/processed_merged_clipped'

    # for f_label, f_raster in zip(fin_label_raster, fin_raster_list):

    #     fin_label = os.path.join(dir_in_label, f_label)
    #     fin_raster = os.path.join(dir_in_raster, f_raster)

    #     print(f'Processing {fin_label}, {fin_raster}')
        
    #     label_overlay(fin_label, fin_raster)

    

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

    
