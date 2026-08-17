# Clustering + Classification

A two-stage machine-learning pipeline on banking transaction data: **discover customer segments** with unsupervised clustering, then **train a classifier to predict** those segments on new records.

<p align="center">
  <img src="screenshots/cluster-pca-2d.png" width="640" alt="Two customer clusters visualized in 2D PCA space with their centroids">
</p>

## Overview

- **Clustering** — [`clustering.ipynb`](./notebooks/clustering.ipynb): K-Means groups customers into segments; PCA projects the high-dimensional features to 2D for the visualization above.
- **Classification** — [`classification.ipynb`](./notebooks/classification.ipynb): Decision Tree and Random Forest models are trained (and tuned) to predict a customer's cluster from their features.

Exploratory analysis of the feature correlations that drive the segmentation:

<p align="center">
  <img src="screenshots/correlation-matrix.png" width="640" alt="Correlation matrix of the transaction features">
</p>

## Tech stack

Python · scikit-learn · pandas · PCA · K-Means · Decision Tree · Random Forest · Matplotlib · seaborn

## Notes

Final project for Dicoding's *Belajar Machine Learning untuk Pemula*. Trained models are saved as `.h5` artifacts alongside the notebooks.
