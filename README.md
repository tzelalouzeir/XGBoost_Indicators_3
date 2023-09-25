
# XGBoost Model Optimization

This repository contains Python code for optimizing the hyperparameters of an XGBoost classifier using the `hyperopt` library. The model is trained to predict trading signals, which are categorized as long, short, or neutral, based on various technical indicators.

## Features

1. **Data Cleaning**: Removes rows with missing values.
2. **Encoding**: Encodes the categorical target column ('Signal').
3. **Hyperparameter Optimization**: Uses Bayesian optimization with `hyperopt` to find the best hyperparameters for the XGBoost classifier.
4. **Model Evaluation**: Evaluates the model's performance on a test set, providing a classification report, confusion matrix, accuracy, and ROC-AUC score.
5. **Feature Importance**: Visualizes the importance of each feature used in the model.

## Installation

Make sure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `xgboost`
- `sklearn`
- `hyperopt`

You can install them using `pip`:

```bash
pip install numpy matplotlib xgboost sklearn hyperopt
```

## Usage

1. Clone this repository.
2. Navigate to the repository's root directory in your terminal.
3. Run the code using the following command:

```bash
python xgb_hyperop.py
```

## Sample Output

After running the code, you will get:

1. A printout of the best hyperparameters discovered during the optimization.
2. A bar chart visualizing the feature importances.
3. A detailed performance report including accuracy, ROC-AUC score, classification report, and confusion matrix.
4. An ROC Curve plot.
