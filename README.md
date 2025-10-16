# SML-Project

## Overview
This repository contains a cross-region housing price study based on the Melbourne dataset. The goal is to compare ridge regression and random forest models under a leave-one-region-out setup to understand how well each approach generalizes across metropolitan areas.
## How to Preprocess Data and Construct Feature
 Make sure Melbourne_housing.csv ang ipynb files in the same folder.
1. open data_preprocessing.ipynb in jupter notebook
   * click run all
   * Wait until all cells finish executing
2. open constract_features.ipynb in jupter notebook
   * click run all
   * Wait until all cells finish executing

## How to Train and Run The Models
1. Verify that processed `dataset.csv` is available in the project root.
2. Execute the ridge evaluation script:
   ```bash
   python cross_region_ridge.py
   ```
3. Execute the random forest evaluation script:
   ```bash
   python cross_region_rf.py
   ```
4. Open light_gbm.ipynb and run all code blocks, the notebook will work the best with GPU acceleration
All scripts and notebooks report cross-validation statistics and hold-out region metrics that you can reference in the project report.
