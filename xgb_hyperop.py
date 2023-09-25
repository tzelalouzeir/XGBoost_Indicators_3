import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, log_loss

# As we mentioned Signal Dataframe is available at ta_mean.py https://github.com/tzelalouzeir/XGBoost_Indicators
# Merge with this code or at ta_mean.py save Signals column to csv
# Available to modification, choose best parameters for your data

# Remove NaN rows
df.dropna(inplace=True)

# Label encode the target column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Signal'])

# Prepare data
X = df.drop(['Signal'], axis=1)
y = y_encoded

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for hyperopt optimization
def objective(space):
    clf = XGBClassifier(
        n_estimators = int(space['n_estimators']),
        max_depth = int(space['max_depth']),
        learning_rate = space['learning_rate'],
        gamma = space['gamma'],
        min_child_weight = int(space['min_child_weight']),
        subsample = space['subsample'],
        colsample_bytree = space['colsample_bytree'],
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    acc = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
    return {'loss': -acc, 'status': STATUS_OK }

# Hyperparameter space
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 1),
    'max_depth': hp.quniform('max_depth', 3, 14, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'gamma': hp.uniform('gamma', 0, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1)
}

# Run optimization
trials = Trials()
best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=50,
                        trials=trials)

# Print the best hyperparameters
print("\nBest hyperparameters:")
for key, value in best_hyperparams.items():
    if key in ['n_estimators', 'max_depth', 'min_child_weight']:
        print(f"{key}: {int(value)}")
    else:
        print(f"{key}: {value}")

# Use best hyperparameters to fit the model
best_clf = XGBClassifier(
    n_estimators=int(best_hyperparams['n_estimators']),
    max_depth=int(best_hyperparams['max_depth']),
    learning_rate=best_hyperparams['learning_rate'],
    gamma=best_hyperparams['gamma'],
    min_child_weight=int(best_hyperparams['min_child_weight']),
    subsample=best_hyperparams['subsample'],
    colsample_bytree=best_hyperparams['colsample_bytree'],
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
best_clf.fit(X_train, y_train)
# # Get feature importances
importances = best_clf.feature_importances_
feature_names = X.columns

# Get and plot feature importances
importances = best_clf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_indices]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), sorted_names, rotation=90)
plt.show()

## performance
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
matrix = confusion_matrix(y_test, y_pred)
probabilities = best_clf.predict_proba(X_test)[:, 1]  # get the probability of the positive class
auc = roc_auc_score(y_test, probabilities)

print(matrix)
print(report)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {auc:.2f}")
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
