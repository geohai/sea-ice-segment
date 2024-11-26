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
    left = [np.inf, 0]
    right = [-np.inf, 0]
    top = [0, -np.inf]
    bottom = [0, np.inf]
    
    lyr = dissolved_shape.GetLayer()

    for feature in lyr:
        geom = feature.GetGeometryRef()
        geom = geom.GetBoundary()  # Get the boundary of the geometry
        ring = geom.GetGeometryRef(0)  # Assuming the geometry is a polygon and we need the outer ring

        points = ring.GetPoints()  # Get all the points of the polygon

        for point in points:
            x, y = point[:2]
            if x < left[0]:
                left = [x, y]
            if x > right[0]:
                right = [x, y]
            if y > top[1]:
                top = [x, y]
            if y < bottom[1]:
                bottom = [x, y]

    # distance from top to right:
    upper_right_line_halfdist = np.sqrt((top[0]-right[0])**2+(top[1]-right[1])**2)/2
    # angle:
    theta = np.arctan2([top[1]-right[1]], [top[0]-right[0]])
    # point at top right
    right_top_halfway = [np.cos(theta)*upper_right_line_halfdist+right[0], np.sin(theta)*upper_right_line_halfdist+right[1]]

    # distance from bottom to left:
    lower_left_line_halfdist = np.sqrt((left[0]-bottom[0])**2+(left[1]-bottom[1])**2)/2
    # angle:
    theta = np.arctan2([left[1]-bottom[1]], [left[0]-bottom[0]])
    # point at bottom left
    left_bottom_halfway = [np.cos(theta)*lower_left_line_halfdist+bottom[0], np.sin(theta)*lower_left_line_halfdist+bottom[1]]
    
    #############################################
    # use points to generate polygons to clip shapes
    #############################################
    
    ################ SE
    pol = ogr.Geometry(ogr.wkbLinearRing)

    for x, y in zip([right_top_halfway[0], right[0], bottom[0], left_bottom_halfway[0], right_top_halfway[0]],
                    [right_top_halfway[1], right[1], bottom[1], left_bottom_halfway[1], right_top_halfway[1]]):
        pol.AddPoint(float(x), float(y))

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(pol)

    ################ clip
    se_clip_fname = os.path.join(dir_out, f'{Path(fname_in).stem}-SE.shp')
    os.system(f'{ogr2ogr} -clipsrc "{poly.ExportToWkt()}" -f "ESRI Shapefile" {se_clip_fname} {fname_in}')

    ################ NW
    pol = ogr.Geometry(ogr.wkbLinearRing)

    for x, y in zip([right_top_halfway[0], left_bottom_halfway[0], left[0], top[0], right_top_halfway[0]],
                    [right_top_halfway[1], left_bottom_halfway[1], left[1], top[1], right_top_halfway[1]]):
        pol.AddPoint(float(x), float(y))

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(pol)

    ################ clip
    nw_clip_fname = os.path.join(dir_out, f'{Path(fname_in).stem}-NW.shp')
    os.system(f'{ogr2ogr} -clipsrc "{poly.ExportToWkt()}" -f "ESRI Shapefile" {nw_clip_fname} {fname_in}')

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

