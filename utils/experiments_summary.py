import argparse
import configparser
import os

from pathlib import Path
import pandas as pd

def main():
    root_dir = os.path.normpath('E:/rafael/data/Extreme_Earth/results/SA')
    
    summary_list = []

    for fname in Path(root_dir).rglob('*metrics*'):

        if 'lightning_logs' not in os.path.normpath(fname).split(os.sep):
            print(fname)
            with open(fname) as fin:
                metrics_string = fin.read()
                # this won't be necessary after fixes
                metrics_string=metrics_string.replace('/n', '\n')
                metrics_string=metrics_string.replace('macro', ' macro')
                metrics_string=metrics_string.replace('weighted', ' weighted')

            # split by space:
            raster_names = [f for f in metrics_string.replace('\n',' ').split(' ') if os.path.isfile(f)]
            for idx in range(len(raster_names)):
                metrics_for_raster = metrics_string.split(raster_names[idx])
                metrics_for_raster = metrics_for_raster[1].split('performance')[1]
                # use only jaccard index:
                metrics_for_raster = metrics_for_raster.split('Jaccard Index')[1]
                iou_micro = float(metrics_for_raster.split('micro: ')[1].split()[0])
                iou_macro = float(metrics_for_raster.split('macro: ')[1].split()[0])
                iou_weighted = float(metrics_for_raster.split('weighted: ')[1].split()[0])

                summary_list.append({'raster': Path(raster_names[idx]).name,
                                     'experiment': str(fname), 
                                     'training_group': os.path.normpath(str(fname).split('exp')[0]).split(os.sep)[-1],
                                     'iou_micro': iou_micro, 
                                     'iou_macro': iou_macro, 
                                     'iou_weighted': iou_weighted
                                    })
        
    df = pd.DataFrame.from_dict(summary_list)
    df.to_csv(os.path.join(root_dir, 'summary_iou.csv'), index=False)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='main_config.ini')

    args = parser.parse_args()

    main()
