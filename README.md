# Sea ice segmentaion using convolutional neural networks

This repository contains scripts that can be used to train convolutional neural network models to segment sea ice using Synthetic Aperture Radar (SAR) images like the following example. 

![Results example](./resources/figure-5-binary-jul-pred.png)

The models are trained with  three-band rasters as input (a), and labels (rasters) with the same projection and dimensions than the input (b). After training, the model generates outputs (c) using full SAR scenes (a) that can be evaluated agains the original labels (d)

# Getting started
## Using a terminal
Clone this repository to your local machine using your tool of choice. Open the [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) (requires a working [Anaconda](https://www.anaconda.com/) installation):

Then, use the prompt to **navigate to the location of the cloned repository**. Install the [environment](env_exported.yml) using the command:  
`conda env create -f env_exported.yml`

Follow the instructions to activate the new environment:  
`conda activate sea-ice-segment`

We have two environment files: 
- [env_exported](env_exported.yml): the environment exported from  Anaconda's history. This should be enough to replicate the results.
- [env_full](env_full.yml): the full environment installed. This includes more information and might be OS dependent. The experiments were executed using Windows 10 Pro for Workstations, mostly using version 21H2. 

## Data and models
We developed the scripts in this repository using the [Extreme Earth](https://doi.org/10.5281/zenodo.4683174) dataset that provides labels (vector polygons) for the interpretation of sea ice conditions on the East coast of Greenland based on twelve Sentinel-1 images. We use [GDAL](https://gdal.org/) to rasterize the labels ([`rasterize.py`](./utils/rasterize.py)). The [dataloaders](datasets.py) are based in [Xarray](https://docs.xarray.dev/en/stable/) and [rioxarray](https://corteva.github.io/rioxarray/stable/).  

Models were developed using [PyTorch](https://pytorch.org/) and trained using [PyTorch Lighnting](https://www.pytorchlightning.ai/).

## Using the code
Use [`main.py`](main.py) to train the models according to a configuration file (several examples in [configs](./configs/)). Evaluate the models using  [`evaluate.py`](evaluate.py) that also requires [configuration files](./configs/).

One of the arguments in the configuration for [`main.py`](main.py) is `fname_csv` that is the file name for a csv files containing raster identification such as the example in csv [example](./resources/EE_IO-poly_type-all.csv)

## References
Please cite us!

The code in this repository was the main source for a sudy published as a [peer-reviewed paper](https://www.tandfonline.com/doi/citedby/10.1080/01431161.2023.2248560) titled **Enhancing sea ice segmentation in Sentinel-1 images with atrous convolutions**. The [preprint](https://arxiv.org/abs/2310.17122) is also available. The bibtex entry for the paper is below:

```bibtex
@ARTICLE{enhancing_sea_ice_2023,
    author = {Pires de Lima, and Vahedi, Behzad and Hughes, Nick and Barrett, Andrew P. and Meier, Walter and Karimzadeh, Morteza},
    title = {Enhancing sea ice segmentation in Sentinel-1 images with atrous convolutions},
    journal = {International Journal of Remote Sensing},
    volume = {44},
    number = {17},
    pages = {5344-5374},
    year = {2023},
    publisher = {Taylor & Francis},
    doi = {10.1080/01431161.2023.2248560},
}
```

Atrous convolution was only part of a series of published studies on machine learning for sea ice. Code in [https://github.com/geohai/sea-ice-binary-ai4seaice/tree/main](https://github.com/geohai/sea-ice-binary-ai4seaice/tree/main) might also be of interest. Other than uncertainty analysis, we talked about:

* [Model Ensemble With Dropout for Uncertainty Estimation in Sea Ice Segmentation Using Sentinel-1 SAR](https://ieeexplore.ieee.org/abstract/document/10312772) ([preprint](https://eartharxiv.org/repository/view/6568/))
* [Comparison of Cross-Entropy, Dice, and Focal Loss for Sea Ice Type Segmentation](https://ieeexplore.ieee.org/abstract/document/10282060) ([preprint](https://arxiv.org/abs/2310.17135))
* [Deep Learning on SAR Imagery: Transfer Learning Versus Randomly Initialized Weights](https://ieeexplore.ieee.org/abstract/document/10281892) ([preprint](https://arxiv.org/abs/2310.17126))

The `bibtex` entry for the publications above is listed below:


```bibtex
@ARTICLE{sea_ice_unc_2023,
  author={Pires de Lima, Rafael and Karimzadeh, Morteza},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Model Ensemble With Dropout for Uncertainty Estimation in Sea Ice Segmentation Using Sentinel-1 SAR}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3331276}
}
```

```bibtex
@INPROCEEDINGS{comparison_sea_ice_2023,
  author={Pires de Lima, Rafael and Vahedi, Behzad and Karimzadeh, Morteza},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Comparison of Cross-Entropy, Dice, and Focal Loss for Sea Ice Type Segmentation}, 
  year={2023},
  pages={145-148},
  address = {Pasadena, CA, USA}
  doi={10.1109/IGARSS52108.2023.10282060}
}
```

```bibtex
@INPROCEEDINGS{tl_sea_ice_2023,
  author={Karimzadeh, Morteza and Pires de Lima, Rafael},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Deep Learning on SAR Imagery: Transfer Learning Versus Randomly Initialized Weights}, 
  year={2023},
  pages={1983-1986},
  address = {Pasadena, CA, USA}
  doi={10.1109/IGARSS52108.2023.10281892}
}
```

