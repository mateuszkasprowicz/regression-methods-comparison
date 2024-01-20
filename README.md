# Comparison of 3 regression methods on 3 different datasets with MLflow

Methods used:
1. linear regression
2. KNN
3. random forest

Datasets used:
1. [Melbourne Housing](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) 
2. [Wine Quality](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
3. [Abalone](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

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
