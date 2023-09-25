
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

![Features](https://github.com/tzelalouzeir/XGBoost_Indicators_3/blob/main/img/features.png)
![ROC](https://github.com/tzelalouzeir/XGBoost_Indicators_3/blob/main/img/roc.png)
![Performance](https://github.com/tzelalouzeir/XGBoost_Indicators_3/blob/main/img/per.PNG)


## Conclusion

Upon evaluating the XGBoost model's performance using the test dataset, we observed the following:

- The confusion matrix shows that the model correctly predicted 218 'long' signals and 239 'short' signals. However, there were 54 'long' signals that were misclassified as 'short' and 54 'short' signals that were misclassified as 'long'.
  
- The precision, recall, and F1-score for both 'long' and 'short' signals are approximately 0.81, indicating a balanced performance between the two classes.
  
- The overall accuracy of the model stands at 81%, which means that the model correctly predicted the trading signals for 81% of the test data.
  
- The ROC-AUC Score is 0.89, which is quite impressive. An ROC-AUC score closer to 1 suggests that the model has good discriminative power between positive and negative classes.

## Related Projects

- [Finding Features with XGBoost](https://github.com/tzelalouzeir/XGBoost_Indicators_2): Training and evaluating an XGBoost classifier on the Bitcoin technical indicators dataset. It aims to predict trading signals (like 'long', 'short', or 'neutral') based on the values of various indicators.
- [Technical Analysis Repository](<https://github.com/tzelalouzeir/XGBoost_Indicators>): This repository fetches 120 days of hourly Bitcoin price data, calculates technical indicators, and analyzes the relations between these indicators.

## 🤝 Let's Connect!
Connect with me on [LinkedIn](https://www.linkedin.com/in/tzelalouzeir/).

For more insights into my work, check out my latest project: [tafou.io](https://tafou.io).

I'm always eager to learn, share, and collaborate. If you have experiences, insights, or thoughts about RL, Prophet, XGBoost, SARIMA, ARIMA, or even simple Linear Regression in the domain of forecasting, please create an issue, drop a comment, or even better, submit a PR! 

_Let's learn and grow together!_ 🌱
