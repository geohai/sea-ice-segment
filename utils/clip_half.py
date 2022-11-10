
import argparse
import configparser
import os
import sys
from pathlib import Path

from osgeo import ogr
import numpy as np

def find_gdal_exe(exe_prog):
    # find the python executer calling this function
    # and search for the gdal program 
    exe_prog = os.path.normpath(os.path.join(str(Path(sys.executable).parent), 
                                'Library', 'bin',
                                f'{exe_prog}.exe'))
    if not os.path.isfile(exe_prog):
        print(f'*****Could not find {exe_prog}.exe')
        return False
    else:
        return exe_prog

def main(fname_in, bound_in):

    fname_in = os.path.normpath(fname_in)
    bound_in = os.path.normpath(bound_in)
    dir_out = os.path.dirname(fname_in)
    temp_boundary_fname = os.path.join(dir_out, 'temp.shp')

    ogr2ogr = find_gdal_exe('ogr2ogr')

    # dissolve
    _command = f'{ogr2ogr} {temp_boundary_fname} {bound_in} '
    _command += f'-dialect sqlite -sql "SELECT ST_Union(geometry) AS geometry FROM {Path(bound_in).stem}"'
    os.system(_command)

    dissolved_shape = ogr.Open(temp_boundary_fname, 1)

    # get the corners
    left = [-np.inf, -np.inf]
    right = [+np.inf, +np.inf]
    bottom = [-np.inf, -np.inf]
    top = [+np.inf, +np.inf]
    
    lyr = dissolved_shape.GetLayer()

    for feature in lyr:
        coords_str = feature.GetGeometryRef().ExportToWkt()
        coords_str = coords_str.replace('MULTIPOLYGON ','').replace('POLYGON ','').replace('(','').replace(')','').split(',')
        coords_str = [val.split(' ') for val in coords_str]
        coords_arr = np.reshape(np.array([float(val) for sublist in coords_str for val in sublist]), (-1,2))

        this_argxmin, this_argymin = np.argmin(coords_arr, axis=0)
        this_argxmax, this_argymax = np.argmax(coords_arr, axis=0)
        
        left_new =   coords_arr[this_argxmin]
        right_new =  coords_arr[this_argxmax]
        bottom_new = coords_arr[this_argymin]
        top_new =    coords_arr[this_argymax]

        left = left_new if left_new[0] > left[0] else left
        right = right_new if right_new[0] < right[0] else right
        bottom = bottom_new if bottom_new[0] > bottom[0] else bottom
        top = top_new if top_new[1] < top[1] else top

    # distance from left to top:
    upper_left_line_halfdist = np.sqrt((top[0]-left[0])**2+(top[1]-left[1])**2)/2
    # angle:
    theta = np.arctan2([top[1]-left[1]], [top[0]-left[0]])
    # point at left top
    left_top_halfway = [np.cos(theta)*upper_left_line_halfdist+left[0], np.sin(theta)*upper_left_line_halfdist+left[1]]

    # distance from bottom to right:
    lower_right_line_halfdist = np.sqrt((right[0]-bottom[0])**2+(right[1]-bottom[1])**2)/2
    # angle:
    theta = np.arctan2([right[1]-bottom[1]], [right[0]-bottom[0]])
    # point at left top
    lower_right_halfway = [np.cos(theta)*lower_right_line_halfdist+bottom[0], np.sin(theta)*lower_right_line_halfdist+bottom[1]]
    
    #############################################
    # use points to generate polygons to clip shapes
    #############################################
    
    ################ NE
    pol = ogr.Geometry(ogr.wkbLinearRing)

    for x, y in zip([left_top_halfway[0], top[0], right[0], lower_right_halfway[0], left_top_halfway[0]],
                    [left_top_halfway[1], top[1], right[1], lower_right_halfway[1], left_top_halfway[1]]):
        pol.AddPoint(float(x), float(y))

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(pol)

    ################ clip
    ne_clip_fname = os.path.join(dir_out, f'{Path(fname_in).stem}-NE.shp')
    os.system(f'{ogr2ogr} -clipsrc "{poly.ExportToWkt()}" -f "ESRI Shapefile" {ne_clip_fname} {fname_in}')

    ################ SW
    pol = ogr.Geometry(ogr.wkbLinearRing)

    for x, y in zip([left_top_halfway[0], left[0], bottom[0], lower_right_halfway[0], left_top_halfway[0]],
                    [left_top_halfway[1], left[1], bottom[1], lower_right_halfway[1], left_top_halfway[1]]):
        pol.AddPoint(float(x), float(y))

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(pol)

    ################ clip
    sw_clip_fname = os.path.join(dir_out, f'{Path(fname_in).stem}-SW.shp')
    os.system(f'{ogr2ogr} -clipsrc "{poly.ExportToWkt()}" -f "ESRI Shapefile" {sw_clip_fname} {fname_in}')

    # remove temporary files
    dissolved_shape = None

    tempfiles = [ff for ff in os.listdir(dir_out) if 'temp' in ff]
    for tempfile in tempfiles:
        if os.path.isfile(os.path.join(dir_out, tempfile)):
            os.remove(os.path.join(dir_out, tempfile))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='utils/clip_half_config.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        fnames_in = [val for val in config['io']['fin_shape_list'].split('\n')]
        bounds_in = [val for val in config['io']['fin_boundary_list'].split('\n')]

        for fname_in, bound_in in zip(fnames_in, bounds_in):
            print(f'Processing {fname_in}')
            main(fname_in, bound_in)
    
    else:
        print('Please provide a valid configuration file.')

