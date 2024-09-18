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


## Citation
Please cite the following paper if you use this repository.

```
@article{Shiri2024,
  author = {Isaac Shiri and Sebastian Balzer and Giovanni Baj and Benedikt Bernhard and Moritz Hundertmark and Adam Bakula and Masaaki Nakase and Daijiro Tomii and Giulia Barbati and Stephan Dobner and Waldo Valenzuela and Axel Rominger and Federico Caobelli and George C. M. Siontis and Jonas Lanz and Thomas Pilgrim and Stephan Windecker and Stefan Stortecky and Christoph Gräni},
  title = {Multi-modality artificial intelligence-based transthyretin amyloid cardiomyopathy detection in patients with severe aortic stenosis},
  journal = {European Journal of Nuclear Medicine and Molecular Imaging},
  year = {2024},
  note = {In press}
}

```
Shiri, I., Balzer, S., Baj, G., Bernhard, B., Hundertmark, M., Bakula, A., Nakase, M., Tomii, D., Barbati, G., Dobner, S., Valenzuela, W., Rominger, A., Caobelli, F., Siontis, G. C. M., Lanz, J., Pilgrim, T., Windecker, S., Stortecky, S., & Gräni, C. Multi-modality artificial intelligence-based transthyretin amyloid cardiomyopathy detection in patients with severe aortic stenosis. European Journal of Nuclear Medicine and Molecular Imaging, (2024) In press.
