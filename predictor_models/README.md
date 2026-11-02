# DN and DM predictor models

This module contains the training and evaluation workflows for the two property
predictors used by the molecular-generation workflow. It is independent of the
JT-VAE fine-tuning code.

## DN predictor

1. dn/01_descriptor_selection.ipynb cleans descriptors, evaluates descriptor
   variability and compares regressors across feature counts.
2. dn/02_gradient_boosting_training.ipynb trains and evaluates the selected
   gradient-boosting DN model, including predicted-versus-reference plots.

## DM predictor

1. dm/01_descriptor_selection.ipynb cleans descriptors and evaluates retained
   descriptor variability.
2. dm/02_ridge_regression_training.ipynb trains and evaluates the selected
   ridge-regression DM model, including predicted-versus-reference plots.

The DN and DM source tables are distributed in the public companion data
archive rather than committed to GitHub.

After running data_setup/install_external_data.py, the DN and DM source tables
are available under data/source_data/property_models/. Run each notebook from
its own dn/ or dm/ directory so that its relative input and output paths resolve
as documented.
