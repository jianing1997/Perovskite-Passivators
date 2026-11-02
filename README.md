# Pareto-guided inverse design of perovskite passivators

This repository is organized by scientific workflow. The DN/DM predictor
training, JT-VAE fine-tuning and manuscript analysis are separated so each part
can be understood and reproduced independently.

The public companion data archive contains the source tables and generated
results used by the predictor notebooks and figure scripts. Large third-party
JT-VAE/ZINC runtime assets are not redistributed.

## Public data download

The source data and optimization results required by the public predictor and
figure workflows are archived on Zenodo:

- **Dataset:** *Source data and optimization results for Pareto-guided inverse design of perovskite passivators*
- **Version:** 1.0.0
- **DOI and download page:** https://doi.org/10.5281/zenodo.23855315

Download `perovskite_passivators_results_source_data_v1.zip` from the Zenodo
record, extract it, and install it as described in the Quick start section.

## Authors and citation

This GitHub release was prepared by **Ning Jia**, with **Yang Pu** and
**Zhe Liu** listed as co-authors for their contributions to the modelling,
code, and released data.

Please cite the public companion dataset as:

> Jia, N., Pu, Y., & Liu, Z. (2026). *Source data and optimization results for
> Pareto-guided inverse design of perovskite passivators* (Version 1.0.0)
> [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23855315

## Repository structure

```text
predictor_models/
  dn/                         DN descriptor selection, training and evaluation
  dm/                         DM descriptor selection, training and evaluation
jtvae_finetuning/
  workflow/                   iterative generate-evaluate-rank-update-fine-tune loop
  launchers/                  manuscript sample-based run configuration
  chem/                       JT-VAE model, data and molecular utilities
analysis_and_figures/
  Figure_*.py                 Python entry points for data-generated key panels
  _shared.py                  shared data paths, styling and plotting helpers
data_setup/                   external data installation utility
runtime_assets/               small tracked runtime files
```

## Where to start

| Task | Start here |
|---|---|
| Train and evaluate the DN predictor | predictor_models/dn/ through predictor_models/README.md |
| Train and evaluate the DM predictor | predictor_models/dm/ through predictor_models/README.md |
| Run Pareto-guided JT-VAE fine-tuning | jtvae_finetuning/launchers/run_paper_workflow.sh |
| Inspect the fine-tuning implementation | jtvae_finetuning/workflow/opt_chem.py |
| Regenerate computational key panels | analysis_and_figures/Figure_*.py |
| Install the public result/source-data release | data_setup/install_external_data.py |

The figure directory is intentionally script-only and is organized by published
figure number. It contains the key panels generated from released numerical data.

## Quick start

1. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate perovskite-passivators
   ```
2. Download and extract `perovskite_passivators_results_source_data_v1.zip`
   from the [Zenodo companion data record](https://doi.org/10.5281/zenodo.23855315).
3. Install the extracted result and source-data folders into the repository:
   ```bash
   python data_setup/install_external_data.py /path/to/external_data_release
   ```
4. Run the full molecular-generation workflow:
   ```bash
   bash jtvae_finetuning/launchers/run_paper_workflow.sh
   ```
   This step additionally requires the JT-VAE/ZINC tensors, pretrained
   checkpoint and surrogate-model files referenced by the launcher. These
   third-party runtime assets are not included in the public data archive.
5. Run a desired figure entry point, for example:
   ```bash
   python analysis_and_figures/Figure_2.py
   ```

## Reproducibility notes

- Each round draws 1,000 monitoring samples. A separate pool of 500 molecules is decoded for selection, and the top 100 are retained.
- The paper setting uses ten rounds, one initial fine-tuning epoch and five fine-tuning epochs in subsequent rounds.

## Data and compatibility notes

- The dependency versions in environment.yml follow the available checkpoint metadata and predictor-model requirements.
- The archived tensor pickles retain the historical weighted_retraining module path. The data loader maps this path to the current jtvae_finetuning namespace without rewriting the archived files.
- Large JT-VAE/ZINC runtime assets and third-party pretrained weights are not
  redistributed with the public release.

## Provenance and licence

The molecular-generation workflow builds on
[GMD-MO-LSO](https://github.com/nafizabeer/GMD-MO-LSO),
[weighted-retraining](https://github.com/cambridge-mlg/weighted-retraining) and the
original [JT-VAE implementation](https://github.com/wengong-jin/icml18-jtnn).
The retained upstream MIT licence is included. The associated article citation
can be added here when it becomes available.
