"""Process S1 data
Assumes SNAP is installed and that can be found in the terminal using 
gpt
uses xml provided in the EE document
This also resamples the data to the desired resolution using gdal
"""
import argparse
import configparser
import os
import shutil
from pathlib import Path

from osgeo import gdal


def main(config):
    
    res = config['params']['res']
    dir_out = os.path.normpath(config['io']['dir_out'])
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    fin_raster_list = [os.path.normpath(val) for val in config['io']['fin_raster_list'].split('\n')]

    _temp_fname_raster = os.path.join(dir_out, '_temp')
    _temp_fname_xml = os.path.join(dir_out, '_temp.xml')

    proj = config['params']['proj']
    
    nanval = -99999

    _placeholder_in = '**place/holder.zip**'
    _placeholder_out = '**place/holder/out**'
    _format = 'ENVI'
    
    with open(config['io']['fin_xml'], 'r') as fin:
        _temp_xml = fin.read()
    
    with open(_temp_fname_xml, 'w') as fout:
        fout.write(_temp_xml)

    for fin in fin_raster_list:

        with open(_temp_fname_xml, 'w') as fxml:
            fxml.write(_temp_xml.replace(_placeholder_in, fin).replace(_placeholder_out, _temp_fname_raster))

        os.system(f'gpt {_temp_fname_xml}')

        # # merge the output into a single raster
        # # (ENVI output is multiple bands and GeoTiff is too slow)
        temp_vrt = os.path.join(dir_out, f'{Path(_temp_fname_raster).stem}.vrt')
        temp_raster = os.path.join(dir_out, f'{Path(_temp_fname_raster).stem}.tif')

        list_of_files = [os.path.join(_temp_fname_raster, val) for val in ['Sigma0_HH_db.img','Sigma0_HV_db.img', 'projectedLocalIncidenceAngle.img']]
        
        vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', separate=True, srcNodata=0, VRTNodata=nanval, outputSRS=proj)
        vrt = gdal.BuildVRT(temp_vrt, list_of_files, options=vrt_options)
        vrt = None

        # create a new raster
        tif = gdal.Translate(temp_raster, temp_vrt)
        tif = None
        os.remove(temp_vrt)

        fout = os.path.join(dir_out, f'{Path(fin).resolve().stem}.tif')
        ds = gdal.Open(temp_raster, 1)
        gdal.Translate(fout, ds, xRes=res, yRes=res)
        ds = None
    
        try:
            shutil.rmtree(os.path.join(dir_out, f'{Path(_temp_fname_raster).stem}'))
        
        except OSError as e:
            print (f"Error: could not remove {e.filename} - {e.strerror}.")
            print ('Retrying...')
            try:
                os.remove(e.filename)
                print (f'{e.filename} removed')
            except OSError as e:
                print (f'{e.filename} not removed!')


    os.remove(temp_raster)
    os.remove(_temp_fname_xml)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='utils/config_s1_proc.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        main(config)

    else:
        print('Please provide valid configuration file')

