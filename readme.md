## Table of contents

- [quick-start](#quick-start)
- [dicom](#dicom)
- [excel](#excel)


## Quick start
    install anaconda or miniconda
    conda create -n <project_name> python=3.9
    pip install poetry
    conda activate <project_name>
    cd /path/to/this/repo
    poetry install

## Dicom
Tag based dicom file converter. 
Tags are not defined yet. Not used yet.

## Excel

### Nice to know
- path_master.py -> holds src and export folder definitions
- data in test folder -> healthy (myocarditis negative, not necessary really healthy)
- data in train folder -> myocarditis positive

#### 1. Pre-processing (to create basic data structure)
- workbook_2_sheets.py  -> extract sheets from workbook and save as separate files
- sheets_2_tables.py -> extract tables from sheets and save as separate files
- cleaner.py -> clean up tables and save in a new folder
- checks.py  -> check if all tables complete and save in a new folder

#### 2. Refinement (more specific data arrangement for faster plotting)
- calculate_accelerations.py -> calculate accelerations from raw data and save in a new folder
- table_condenser.py -> focus on specific data and save in a same folder as acceleration results
- table_merger.py -> merge tables and save in a new folder

#### 3. Analyze (ce plots)
- use jupyter notebook (load the data into the RAM for faster plotting iterations)
