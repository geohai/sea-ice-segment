# STOP!! Before continuing, make sure the original README is followed and the environment is set up

# Getting started

Make sure all your processed image files (L-band and C-band, for instance) are in the same folder (e.g. denoised_resampled). They should have two channels, HH and HV.

Make sure your label files (.shp) are in the same folder (e.g. Sea_Ice-labels).

The following paths were used for my local machine, for your reference:

- Raster files: D:/jema2085/data/Belgica_Bank/denoised_resampled
- Shp files: D:/jema2085/data/Belgica_Bank/MET_BB_Polar_shapefiles/Belgica_Bank-20240506T141908Z-001/Belgica_Bank/Sea_Ice-labels
- Rasterized label files (after running rasterize.py): D:/jema2085/data/Belgica_Bank/labels_rasterized/{raterized type}
- Excel files with training paths: D:/jema2085/data/Belgica_Bank/results
- Training outputs: D:/jema2085/data/Belgica_Bank/results/{rasterized type}/{training split}
- Testing results: D:/jema2085/data/Belgica_Bank/results/{rasterized type}/{training split}
- Averaged testing results (classification report + confusion matrix): D:/jema2085/data/Belgica_Bank/results/{rasterized type}

# Clip the images in half (if needed):

- Run `python utils/clip_half_BB.py -c utils/clip_half_config_BB.ini` (e.g. rasterized type == SA)

# Rasterize the label files:

- Run `python utils/raterize.py -c utils/config-MET-{rasterized type}-dual.ini` (e.g. rasterized type == SA)

# 1. Run with your first image type (e.g. L-band)

## Train the model

- Run `python main_single.py -c configs/{rasterized type}_single_{image type}.ini` (e.g. rasterized type == SA, image type == alos2)

## Test the model

- Run `python evaluate_single.py -c configs/eval_{rasterized type}_{image type}.ini` (e.g. rasterized type == SA, image type == alos2)

## Generate an averaged classification report and confusion matrix

- Run `python generate_avg_metrics.py -c configs/metrics_{rasterized type}_single_{image type}.ini` (e.g. rasterized type == SA, image type == alos2)

# 2. Run with your second image type (e.g. C-band)

## Train the model

- Run `python main_single.py -c configs/{rasterized type}_single_{image type}.ini` (e.g. rasterized type == SA, image type == s1)

## Test the model

- Run `python evaluate_single.py -c configs/eval_{rasterized type}_{image type}.ini` (e.g. rasterized type == SA, image type == s1)

## Generate an averaged classification report and confusion matrix

- Run `python generate_avg_metrics.py -c configs/metrics_{rasterized type}_single_{image type}.ini` (e.g. rasterized type == SA, image type == s1)

# 3. Run with both your images, early fusion

## Train/test the model. This will do both training and testing

- Run `python main_dual_early.py -c configs/{rasterized type}_dual_early.ini` (e.g. rasterized type == SA)

## Generate an averaged classification report and confusion matrix

- Run `python generate_avg_metrics.py -c configs/metrics_{rasterized type}_dual_early.ini` (e.g. rasterized type == SA)

# 4. Run with both your images, mid fusion

## Train/test the model. This will do both training and testing

- Run `python main_dual_mid.py -c configs/{rasterized type}_dual_mid.ini` (e.g. rasterized type == SA)

## Generate an averaged classification report and confusion matrix. There should be three versions that can be generated,

## one testing with both images, one testing with only the first image type, and one testing with only the second image type

- Run `python generate_avg_metrics.py -c configs/metrics_{rasterized type}_dual_mid.ini` (e.g. rasterized type == SA)
- Run `python generate_avg_metrics.py -c configs/metrics_{rasterized type}_dual_mid_{image type}.ini` (e.g. rasterized type == SA, image type == alos2)
- Run `python generate_avg_metrics.py -c configs/metrics_{rasterized type}_dual_mid_{image type}.ini` (e.g. rasterized type == SA, image type == s1)
