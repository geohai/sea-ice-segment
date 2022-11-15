"""Script to generate figures for the weights of the trained models
Reads a trained model and creates the image of the weights for the first layer
modified from https://github.com/raplima/tl-thin-sections/blob/main/weights_figures.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return fig


dir_out = "E:/rafael/data/Extreme_Earth/results/v3/SD_wland-nopretrain/all - random initialization/experiment-notpretrained3"

load_weights_path = os.path.join(dir_out, 'best_weights.ckpt')

print(load_weights_path)
ckpt = torch.load(load_weights_path)
conv1_weigths = ckpt['state_dict']['streams.0.conv1.weight']
conv1_grid = make_grid(conv1_weigths, normalize=True, padding=1)
fig = show(conv1_grid)

fig.savefig(os.path.join(dir_out,
                            f'{os.path.basename(load_weights_path)}.png'))