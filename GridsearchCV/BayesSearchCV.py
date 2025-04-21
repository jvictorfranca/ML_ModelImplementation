from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
    StratifiedKFold
from skopt import BayesSearchCV

from skopt.space import Real, Categorical, Integer

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

titanic = pd.read_pickle('Datasets/pickle/titanic.pkl')
X = titanic.drop(columns='survived')
y=titanic.survived

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(1, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'bootstrap': Categorical([True, False]),
    'criterion': Categorical(['gini', 'entropy']),
    'ccp_alpha': Real(0, 0.05)
}

n_iter=20
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=n_iter,  # Número de iterações
    cv=5,       # 5-Fold Cross-Validation
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print(f"Melhores hiperparâmetros: {bayes_search.best_params_}")
print(f"\nAUC média na validação cruzada: {bayes_search.best_score_:.2%}")

final_clf = bayes_search.best_estimator_

bayes_test_score = final_clf.score(X_test, y_test)
bayes_roc = roc_auc_score(y_test, final_clf.predict_proba(X_test)[:,1])
bayes_gini = bayes_roc*2-1
print(f"Gini do bayesian search no teste: {bayes_gini:.4f}")

resultados_bayes = pd.DataFrame(bayes_search.cv_results_)
resultados_bayes['gini'] = resultados_bayes.mean_test_score*2-1