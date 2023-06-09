import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings

from utils.print_cv_values import print_results

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('CrossValidation/sets_titanic/train_features.csv')
tr_labels = pd.read_csv('CrossValidation/sets_titanic/train_labels.csv', header=None)

lr = LogisticRegression()

parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print(cv.best_params_)
print(cv.cv_results_)
print(cv.best_estimator_)
print_results(cv)

joblib.dump(cv.best_estimator_, 'CrossValidation/best_estimators/LR_model.pkl')