from sklearn.ensemble import GradientBoostingRegressor

import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

np.random.seed(2360873)

# Gerar 1000 valores sequenciais para X

x = np.linspace(0, 1, 1000)

X = [[value] for value in x]
# Definindo os parâmetros da parábola
a = 0
b = 10
c = -10

# Gerar uma relação quadrática com ruído
y = a + b * x + c * x**2 + np.random.normal(loc=0, scale=.3, size=len(x))**3

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 3, 5, 10],
    'learning_rate': [0.1, 0.4],
#    'ccp_alpha': [0.1, 0.01, 0.001, 0.001, 0.0001, 0.00001]
    'ccp_alpha': [0.0001, 0.00001]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=10, verbose=0)

grid_search.fit(X_train, y_train)


modelo = grid_search.best_estimator_

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(modelo)
print(best_params)
print(best_score)


best_tree = GradientBoostingRegressor(**best_params)
best_tree.fit(X_train, y_train)

y_pred = best_tree.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-quadrado na base de testes:", r2)

