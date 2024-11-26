import argparse
import configparser
import os
from pathlib import Path

import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

def parse_metrics(lines):
    metrics = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    averages = {'support': [], 'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': []}, 'weighted avg': {'precision': [], 'recall': [], 'f1-score': []}}
    jaccard = {'micro': [], 'macro': [], 'weighted': []}
    confusion_matrices = []
    
    cm_start = False
    for line in lines:
        if 'performance' in line:
            cm_start = False
            continue

        if '0.0 ' in line or '1.0 ' in line or '2.0 ' in line or '3.0 ' in line:
            values = line.split()
            metrics['precision'].append(float(values[1]))
            metrics['recall'].append(float(values[2]))
            metrics['f1-score'].append(float(values[3]))
            metrics['support'].append(int(values[4]))

        if 'accuracy' in line:
            values = line.split()
            averages['accuracy'].append(float(values[1]))
            averages['support'].append(float(values[2]))

        if 'macro avg' in line:
            values = line.split()
            averages['macro avg']['precision'].append(float(values[2]))
            averages['macro avg']['recall'].append(float(values[3]))
            averages['macro avg']['f1-score'].append(float(values[4]))

        if 'weighted avg' in line:
            values = line.split()
            averages['weighted avg']['precision'].append(float(values[2]))
            averages['weighted avg']['recall'].append(float(values[3]))
            averages['weighted avg']['f1-score'].append(float(values[4]))
        
        if cm_start and list(map(float, line.strip().split())) != []:
            confusion_matrices.append(list(map(float, line.strip().split())))
        
        if 'Confusion Matrix' in line:
            cm_start = True
            continue

        if 'micro:' in line or 'macro:' in line or 'weighted:' in line:
            key, value = line.split(':')
            jaccard[key.strip()].append(float(value.strip()))
    
    return metrics, averages, jaccard, confusion_matrices

def weighted_avg(metrics, key):
    total_support = np.array([np.sum(metrics['support'][i::4]) for i in range(4)])
    unweighted_key = np.array(metrics[key]) * np.array(metrics['support'])
    weighted_sum = [np.sum(unweighted_key[i::4]) for i in range(4)]
    return np.round(weighted_sum / total_support, 2)
    
def macro_avg(averages, key):
    if key == 'accuracy':
        return np.round(np.sum(np.array(averages[key]) * np.array(averages['support']))/np.sum(averages['support']), 2)
    if key == 'macro avg':
        avg = {}
        for k in averages:
            avg[k] = np.mean(averages[k])
        return avg
    else:
        avg = {}
        for k in averages[key]:
            avg[k] = np.round(np.sum(np.array(averages[key][k]) * np.array(averages['support']))/np.sum(averages['support']), 2)
        return avg

def weighted_jaccard(jaccard, support):
    avg = {}
    for key in jaccard:
        avg[key] = np.round(np.sum(np.array(jaccard[key]) * support)/np.sum(support), 2)
    return avg

def combine_cm(confusion_matrix, support):
    multiplied_cm = []
    avg_combined_cm = []
    for i in range(len(confusion_matrix)):
        multiplied_cm.append(np.array(confusion_matrix[i]) * support[i])
    for i in range(4):
        combined_cm = [np.sum(multiplied_cm[i::4], axis=0) for i in range(4)]
        combined_support = [np.sum(support[i::4], axis=0) for i in range(4)]
    for i in range(4):
        avg_combined_cm.append(combined_cm[i] / combined_support[i])
    return avg_combined_cm

def generate_avg_metrics(config):
    metrics_path = os.path.normpath(config['io']['metrics_path'])
    conf_path = os.path.normpath(config['io']['conf_path'])

    with open(metrics_path, 'r') as f:
        lines = f.readlines()

    metrics, averages, jaccard, confusion_matrix = parse_metrics(lines[0:len(lines)])

    combined_metrics = {}
    combined_metrics['precision'] = weighted_avg(metrics, 'precision')
    combined_metrics['recall'] = weighted_avg(metrics, 'recall')
    combined_metrics['f1-score'] = weighted_avg(metrics, 'f1-score')
    combined_metrics['support'] =  np.array([np.sum(metrics['support'][i::4]) for i in range(4)])
    
    macro_metrics = {}
    macro_metrics['accuracy'] = macro_avg(averages, 'accuracy')
    macro_metrics['macro avg'] = macro_avg(combined_metrics, 'macro avg')
    macro_metrics['weighted avg'] = macro_avg(averages, 'weighted avg')

    combined_jaccard = {}
    combined_jaccard = weighted_jaccard(jaccard, averages['support'])

    combined_cm = combine_cm(confusion_matrix, metrics['support'])
    
    with open(metrics_path, 'a', encoding='utf-8') as outfile:
        outfile.write('-' * 80 + '\n')  # Write 80 hyphens
        outfile.write(f'Averaged performance \n')
        outfile.write('              precision    recall  f1-score   support\n\n')
        for i, support in enumerate(combined_metrics['support']):
            outfile.write(f"         {i}.0       {combined_metrics['precision'][i]:.2f}      {combined_metrics['recall'][i]:.2f}      {combined_metrics['f1-score'][i]:.2f}    {support}\n")
        outfile.write('\n')
        outfile.write(f"    accuracy                           {macro_metrics['accuracy']:.2f}   {np.sum(averages['support'])}\n")
        outfile.write(f"   macro avg       {macro_metrics['macro avg']['precision']:.2f}      {macro_metrics['macro avg']['recall']:.2f}      {macro_metrics['macro avg']['f1-score']:.2f}   {np.sum(averages['support'])}\n")
        outfile.write(f"weighted avg       {macro_metrics['weighted avg']['precision']:.2f}      {macro_metrics['weighted avg']['recall']:.2f}      {macro_metrics['weighted avg']['f1-score']:.2f}   {np.sum(averages['support'])}\n\n")

        outfile.write(f'Jaccard Index: \n')
        for key, value in combined_jaccard.items():
            outfile.write(f'{key}: {value:.2f}\n')

        outfile.write('\n')
        outfile.write(f'Confusion Matrix: \n      ')
        for row in combined_cm:
            outfile.write('      '.join(f'{value:.2f}' for value in row) + '\n      ')
        
        outfile.write('\n\n')

        plt.figure(figsize=(8, 6))
        # Replace with different class names if needed
        sns.heatmap(combined_cm, annot=True, fmt='.2f', cmap='viridis', 
                    vmin=0, vmax=1,
                    xticklabels=['New Ice/Nilas', 'Young Ice', 'First Year Ice', 'Old Ice'],
                    yticklabels=['New Ice/Nilas', 'Young Ice', 'First Year Ice', 'Old Ice'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(conf_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        generate_avg_metrics(config)
    
    else:
        print('Please provide a valid configuration file.')
