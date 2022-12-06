import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def make_metric_summary():

    root_dir = os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/SD_wland-nopretrain/')
    
    summary_list = []

    if os.path.isfile(os.path.join(root_dir, 'summary_raster_metrics.csv')):
        os.remove(os.path.join(root_dir, 'summary_raster_metrics.csv'))

    for fname in Path(root_dir).rglob('*metrics*'):

        if ('lightning_logs' not in os.path.normpath(fname).split(os.sep)) and ('summary' not in os.path.normpath(fname)):
            print(fname)
            with open(fname) as fin:
                metrics_string = fin.read()

            # split by space:
            raster_compars = [f for f in metrics_string.split('\n') if f.startswith('2018')]
            for idx in range(len(raster_compars)):
                metrics_for_raster = metrics_string.split(raster_compars[idx])
                metrics_for_raster = metrics_for_raster[1].split('performance')[0]
                # accuracy:
                acc = float(metrics_for_raster.split('accuracy')[1].split()[0])
                # F1:
                f1_macro = float(metrics_for_raster.split('\n   macro avg')[1].split()[2])
                f1_weighted = float(metrics_for_raster.split('weighted avg')[1].split()[2])
                #  jaccard index:
                metrics_for_raster = metrics_for_raster.split('Jaccard Index')[1]
                iou_micro = float(metrics_for_raster.split('micro: ')[1].split()[0])
                iou_macro = float(metrics_for_raster.split('macro: ')[1].split()[0])
                iou_weighted = float(metrics_for_raster.split('weighted: ')[1].split()[0])

                summary_list.append({'raster': raster_compars[idx].split(' performance')[0],
                                     'experiment': str(fname), 
                                     'experiment_group': os.path.normpath(str(fname).split('exp')[0]).split(os.sep)[-1],
                                     'accuracy': acc, 
                                     'f1_macro': f1_macro,
                                     'f1_weighted': f1_weighted,
                                     'iou_micro': iou_micro, 
                                     'iou_macro': iou_macro, 
                                     'iou_weighted': iou_weighted
                                    })
        
    df = pd.DataFrame.from_dict(summary_list)
    df.to_csv(os.path.join(root_dir, 'summary_raster_metrics.csv'), index=False)

    # create a display plot and save testing metrics summaries
    df['Experiment Group'] = df['experiment_group']
    df['Scene'] = pd.to_datetime(df['raster'].str.split(expand=True)[0]).dt.month_name().str.slice(stop=3) 
    df.loc[df['raster'].str.split(expand=True)[2].str.split('-', expand=True)[1]=='SW','Scene'] = df['Scene'] + '-SW'
    df.loc[df['raster'].str.split(expand=True)[2].str.split('-', expand=True)[1]=='NE','Scene'] = df['Scene'] + '-NE'
    df.loc[df['Scene'].isin(['Jan', 'Jul']), 'Scene'] = df['Scene'] + ' (test)'
    # remove training and validation scenes
    df = df.drop(df.index[(df['Scene'].isin(['Feb-SW', 'Feb-NE', 'Mar', 'Apr', 'May', 'Jun-SW', 'Jun-NE', 'Aug-SW', 'Aug-NE', 'Sep', 'Oct', 'Nov', 'Dec-SW', 'Dec-NE'])) & (df['Experiment Group']=='all')], axis=0)
    df = df.drop(df.index[(df['Scene'].isin(['Feb-SW', 'Feb-NE', 'Mar', 'Oct', 'Nov', 'Dec-SW', 'Dec-NE'])) & (df['Experiment Group']=='freeze')], axis=0)
    df = df.drop(df.index[(df['Scene'].isin(['Apr', 'May', 'Jun-SW', 'Jun-NE', 'Aug-SW', 'Aug-NE', 'Sep'])) & (df['Experiment Group']=='melt')], axis=0)
    
    df = df.sort_values('raster')
    # save all test
    df.to_csv(os.path.join(root_dir, 'summary_test_raster_metrics.csv'), index=False)

    # group and save aggregate statistics
    agg_func_math = {'f1_weighted': ['median', 'min', 'max'],
                     'iou_weighted': ['median', 'min', 'max'],
                     }

    df.groupby(['Experiment Group', 'Scene', 'raster']).agg(agg_func_math).sort_values(['Experiment Group', 'raster']).to_csv(os.path.join(root_dir, 'summary_grouped_test_raster_metrics.csv'))

    with sns.axes_style("whitegrid"):    
        fig, ax = plt.subplots(figsize=(8,6), ncols=2)
        sns.barplot(x="f1_weighted", y="Scene",
                    hue='Experiment Group',
                    estimator=np.median,
                    errwidth=0.8,
                    capsize=0.2,
                    dodge=True,
                    ax = ax[0],
                    data=df)
       
        ax[0].set_xlabel('F1 (weighted)')
        ax[0].yaxis.grid(True)
        ax[0].set_xlim(left=0.7)

        g=sns.barplot(x="iou_weighted", y="Scene",
                    hue='Experiment Group',
                    estimator=np.median,
                    errwidth=0.8,
                    capsize=0.2,
                    dodge=True,
                    ax = ax[1],
                    data=df) 

        ax[1].set_xlabel('IoU (weighted)')
        ax[1].yaxis.grid(True)

        leg = g.get_legend()
        leg.remove()
        ax[1].yaxis.set_ticklabels([])
        ax[1].set_ylabel('')
        ax[1].set_xlim(left=0.7)

        fig.tight_layout()
        fig.savefig(os.path.join(root_dir, 'full-test-f1_iou_weighted.pdf'))


def full_loss_decay():

    root_dir = os.path.normpath('E:/rafael/data/Extreme_Earth/results/v3/SD_wland-nopretrain')
    
    df_list = []

    for fname in Path(root_dir).rglob('*metrics*'):

        if set(['lightning_logs', 'version_1']).issubset(set(os.path.normpath(fname).split(os.sep))):
            print(fname)

            df_list.append(pd.read_csv(fname))
            df_list[-1]['experiment_full'] = fname
            df_list[-1]['experiment_number'] = os.path.normpath(fname).split(os.sep)[-4]
            df_list[-1]['Experiment Group'] = os.path.normpath(fname).split(os.sep)[-5]

    df = pd.concat(df_list).reset_index(drop=True)
    
    # create dset split column and remove nan vals
    df['Set'] = np.nan
    for dset in ['val_loss', 'train_loss_epoch']:
        df.loc[~df[dset].isnull(), 'Set'] = dset.split('_')[0]
    
    df = df[~df['Set'].isnull()].dropna(axis=1, how='all').drop('step', axis=1)

    cols = [i for i in df.columns if i not in ['Set', 'experiment', 'epoch', 'experiment_full', 'experiment_number', 'Experiment Group']]

    df_long = df.melt(id_vars=['epoch', 'Set', 'experiment_full', 'experiment_number', 'Experiment Group'], value_vars=cols).dropna(axis=0, how='any')

    # generate plots for each metric
    for metric in ['Loss', 'IoU (macro)', 'F1 (macro)']:
        
        df_long_select = df_long[df_long['variable'].str.lower().str.contains(metric.split(' ')[0].lower())].copy()
        df_long_select['variable'] = df_long_select['variable'].str.split('_', expand=True)[0]

        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(4,4))
            sns.lineplot(x="epoch", y="value",
                         style="Set",
                         hue='Experiment Group',
                         ax = ax,
                         estimator=np.median,
                         data=df_long_select)

            if metric == 'Loss':
                ax.set(yscale="log")                        
            
            else:
                ax.set_ylim(bottom=0.0)
                ax.set_ylim(top=0.8)                        

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(os.path.join(root_dir, f'{metric}.pdf'))


if __name__ == '__main__':
    
    #full_loss_decay()
    make_metric_summary()