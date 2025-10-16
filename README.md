# SML-Project

## Overview
This repository contains a cross-region housing price study based on the Melbourne dataset. The goal is to compare ridge regression and random forest models under a leave-one-region-out setup to understand how well each approach generalizes across metropolitan areas.

## How to Run
1. Verify that `dataset.csv` is available in the project root.
2. Execute the ridge evaluation script:
   ```bash
   python cross_region_ridge.py
   ```
3. Execute the random forest evaluation script:
   ```bash
   python cross_region_rf.py
   ```
Both scripts report cross-validation statistics and hold-out region metrics that you can reference in the project report.
