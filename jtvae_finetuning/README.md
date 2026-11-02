# JT-VAE fine-tuning and Pareto optimization

This module contains the molecular generator and the complete iterative
fine-tuning workflow. DN and DM predictors are loaded as fixed surrogate models;
their training notebooks are located in predictor_models/.

## Main entry point

- launchers/run_paper_workflow.sh: manuscript DN-DM run configuration.
- workflow/opt_chem.py: generate, evaluate, Pareto-rank, update and fine-tune loop.
- utils.py: Pareto ranking and sample-weight utilities.

## Model implementation

- chem/: dataset wrappers, property evaluation and PyTorch-Lightning JT-VAE.
- chem/jtnn/: junction-tree encoder, decoder and molecular-tree operations.

The full workflow requires the tensor datasets, pretrained checkpoint and
surrogate-model files referenced by the launcher. These large third-party
runtime assets are not redistributed in the public companion data archive.
