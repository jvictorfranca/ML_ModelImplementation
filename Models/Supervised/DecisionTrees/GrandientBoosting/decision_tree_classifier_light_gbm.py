import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
import lightgbm as lgb

X_train = pd.read_pickle('Datasets/pickle/activities_X_train.pkl')
y_train = pd.read_pickle('Datasets/pickle/activities_y_train.pkl')
X_test = pd.read_pickle('Datasets/pickle/activities_X_test.pkl')
y_test = pd.read_pickle('Datasets/pickle/activities_y_test.pkl')

# Identificar e remover colunas duplicadas

duplicated_columns = X_train.columns[X_train.columns.duplicated()]
print(f'Colunas duplicadas: {duplicated_columns}')
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

# Ajuste de Índice
X_train.set_index('subject', append=True, inplace=True)
X_test.set_index('subject', append=True, inplace=True)

HAR_train = pd.concat([X_train.reset_index(), y_train], axis=1).set_index(['level_0', 'subject'])
print(HAR_train.columns)

np.random.seed(1729)
arvore = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=2)
arvore.fit(X_train, y_train)

importancias = pd.DataFrame(arvore.feature_importances_, index=X_train.columns, columns=['importancia'])
top_10_variaveis = importancias.sort_values(by='importancia', ascending=False)[:10]
print(f'Top 10 variaveis: {top_10_variaveis}')

# Selecionar as 20 variáveis com maior importância
variaveis = importancias.nlargest(20, 'importancia').index.tolist()

y = y_train['label'].cat.codes

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1729)

scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

modelo = lgb.LGBMClassifier(objective='multiclass', random_state=1729)

param_grid = {
    'num_leaves': [31],
    'max_depth': [3, 10],
    'learning_rate': [0.05, 0.2],
    'n_estimators': [5, 11]
}

grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=1)
grid_search.fit(X_train[variaveis], y)

print(f'Melhores Parâmetros: {grid_search.best_params_}')

resultados_cv = pd.DataFrame(grid_search.cv_results_)

pred_test = pd.Series(grid_search.best_estimator_.predict(X_test[variaveis]))

print(pd.crosstab(pred_test, y_test.label))

acurácia = (pred_test == y_test.label.cat.codes).sum()/len(y_test)

print(f'acurácia = {acurácia:.2%}')