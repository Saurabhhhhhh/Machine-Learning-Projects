# House Price Predictor🏠

## Overview

- This project includes a machine learning model to predict house prices and a Tkinter-based GUI for user interaction.

## Machine Learning Model

- Data: Loaded from `data.csv`.
- Preprocessing:
  - Train-test split using stratified sampling.
  - Handling missing values with SimpleImputer.
  - Scaling with StandardScaler.
- Model: Random Forest Regressor.
- Performance: Evaluated using RMSE and cross-validation.
- Saving: Model saved as `model.joblib`.

## Tkinter GUI

- Input: Fields for house features.
- Output: Predicted house price.
- Action: Predict button triggers the model.

## Files

- `data.csv`: Housing data for training.
- `model.joblib`: Saved model.
- `main.py`: Tkinter GUI code.
- `model.ipynb`: Jupyter Notebook of Model.
- `model.py`: Python Script of Model.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- Tkinter
- joblib

## Installation

1. Clone repo: `git clone https://github.com/Saurabhhhhhh/Machine-Learning-Projects.git`
2. Navigate to the project directory: `cd '.\Machine-Learning-Projects\House Price Predictor\'`
3. Install packages: `pip install -r requirements.txt`

## Run

- Model: `model.ipyb`
- Tkinter GUI: `main.py`

## Contributing

- Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
