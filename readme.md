# ML Pipeline for tabular data <!-- omit in toc -->

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Functionalities](#functionalities)
- [Configuration](#configuration)
- [Run](#run)

## Installation

```bash
    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
```

## Functionalities

This pipeline for tabular data offers the following functionalities:

- automatic clean-up
- splitting of dataset for any number of desired seeds or bootstraps
- imputation of missing data and data normalisation
- oversampling (if desired)
- ability to run multiple feature selection strategies which can be configured step-by-step
- verification of these strategies using one or more models
- explainability
  
## Configuration

Make sure to configure everything needed for your experiments in the **config.yaml** file.\
Most important is the target_label, input_file and the label_as_index (if available).\
Other noteworthy entries in the config file:

- meta:
  - workers: set according to your machine
- impute:
  - method: method to use for imputation of missing values
- data_split:
  - n_seeds: number of data split seeds to run
  - test_frac: fraction of dataset to use for testing
- selection:
  - scoring: the metric to use for training during selection and verification
  - jobs: each list defines a job of desired feature selection steps and normalisation
- verification:
  - models: models to train and test
  - param_grids: parameter grids for GridSearchCV

## Run

After the config file is set up properly, you can run the pipeline using:

```bash
python3 main.py
```

Computation progress is saved after each seed/bootstrap and will not be recomputed unless the meta.overwrite flag is set to True.
