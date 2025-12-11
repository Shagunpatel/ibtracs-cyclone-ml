# IBTrACS Cyclone ML Project

This project uses the IBTrACS last 3 years dataset to predict tropical cyclone
wind speed (`USA_WIND`) based on basic track information.

## Dataset

Download the dataset from:

- https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv

Save it as:

- `data/ibtracs.last3years.list.v04r01.csv`

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows (PowerShell):
   venv\Scripts\Activate.ps1
   # Mac/Linux:
   source venv/bin/activate

## Install dependencies:
pip install -r requirements.txt

## Running the code
From the project root:

# Training
python train.py

This will train a Random Forest model and save it to models/random_forest_usa_wind.pkl.

# Testing
python test.py

This will load the saved model and evaluate it on the test split, printing MAE and R².

# Visualizations
python visualize.py

This will generate:

figures/wind_distribution.png
figures/feature_importance.png
