# Comparison of 3 regression methods on 3 different datasets

## How to set up the environment?

### Venv + pip

1. Create virtualenv: `make virtual-env`
2. Activate virtualenv: `source .env/bin/activate`
3. Install requirements: `make pip-install-requirements`

### Conda

1. Create Conda environment: `make conda-environment`
2. Activate Conda environment: `conda activate regression-methods-comparison`

## How to download the datasets?

1. You will need the Kaggle API credentials for that. Read more here: https://github.com/Kaggle/kaggle-api#api-credentials
2. Download the datasets: `make download-datasets`
