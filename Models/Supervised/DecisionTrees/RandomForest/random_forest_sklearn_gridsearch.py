import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import patsy
import time

titanic = sns.load_dataset('titanic')

titanic.head()


# Transforms the notation in a design matrix . Creates dummies automatically.
y, X = patsy.dmatrices('survived ~ pclass + sex + age + sibsp + parch + fare + embarked', data=titanic, return_type="dataframe")

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2360873)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


param_grid = {'n_estimators': [100,200], 'max_features': range(1, 11)}

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, 
                           param_grid=param_grid, 
                           scoring='roc_auc', 
                           cv=4, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train.values.ravel()) 

# Print the best parameters and the best score
print(grid_search)
print(grid_search.best_params_)
print(grid_search.best_score_)
tempo_fim = time.time()

melhor_modelo = grid_search.best_estimator_

print(melhor_modelo.predict([[1,1,1,1,1,1,1,1,1]]))