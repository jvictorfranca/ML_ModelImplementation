import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Semente aleatória para reprodutibilidade (sacou o eastereg?)
np.random.seed(2360873)

# Gerar 1000 valores sequenciais para X
x = np.linspace(0, 1, 1000)

# Definindo os parâmetros da parábola
a = 0
b = 10
c = -10

# Gerar uma relação quadrática com ruído
y = a + b * x + c * x**2 + np.random.normal(loc=0, scale=.3, size=len(x))**3

# Criar o data frame
df = pd.DataFrame({'x': x, 'y': y})

x = df[['x']]
y = df[['y']]

# Treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=30, ccp_alpha=0)
tree.fit(df[['x']], df['y'])

path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
sns.scatterplot(x = ccp_alphas, y = impurities)
len(ccp_alphas)

grid = tree.cost_complexity_pruning_path(X_train, y_train)

print(ccp_alphas)

param_grid  = {'ccp_alpha': ccp_alphas[::10]}

grid_search = GridSearchCV(estimator=tree, 
                          param_grid=param_grid,
                          cv=5,  # 5-fold cross-validation
                          scoring='neg_mean_squared_error')  # Metrica de avaliação

# Treinando o modelo com o grid search
grid_search.fit(X_train, y_train)

best_score = grid_search.best_score_
print(best_score)
best_params = grid_search.best_params_
print(best_params)

best_tree = DecisionTreeRegressor(**best_params, max_depth=30)
best_tree.fit(X_train, y_train)

y_pred = best_tree.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-quadrado na base de testes:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Comparação entre Valores Reais e Previstos')
plt.show()
