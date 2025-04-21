from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, classification_report
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import lightgbm as lgb
import pandas as pd

X_train = pd.read_pickle('Datasets/pickle/activities_X_train.pkl')
y_train = pd.read_pickle('Datasets/pickle/activities_y_train.pkl')['label']
X_test  = pd.read_pickle('Datasets/pickle/activities_X_test.pkl')
y_test  = pd.read_pickle('Datasets/pickle/activities_y_test.pkl')['label']

níveis = y_test.cat.categories
print(níveis)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

param_space = {
    'n_estimators': Integer(50, 500),  # Número de árvores
    'max_depth': Integer(3, 15),       # Profundidade máxima das árvores
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),  # Taxa de aprendizado
    'num_leaves': Integer(20, 100),    # Número máximo de folhas
    'min_child_samples': Integer(10, 100),  # Número mínimo de amostras por folha
    'subsample': Real(0.5, 1.0),       # Subamostragem de dados
    'colsample_bytree': Real(0.5, 1.0),  # Subamostragem de features
    'reg_alpha': Real(0, 1),           # Regularização L1
    'reg_lambda': Real(0, 1),          # Regularização L2
    'boosting_type': Categorical(['gbdt', 'dart'])  # Tipo de boosting
}

lgb_model = lgb.LGBMClassifier(random_state=2244000, verbose=-1)

bayes_search = BayesSearchCV(
    estimator=lgb_model,
    search_spaces=param_space,
    n_iter=5,  # Número de iterações
    cv=2,       # Número de folds na validação cruzada
    scoring='accuracy',
    n_jobs=-1,  # Usar todos os núcleos do processador
    verbose=1,
    random_state=2244000
)

bayes_search.fit(X_train, y_train)

print("Melhores hiperparâmetros:", bayes_search.best_params_)

pred = bayes_search.predict(X_test)

cm = confusion_matrix(y_test, pred)
print("Matriz de Confusão:")
print(cm)
print("\nRelatório de Classificação:")
print(classification_report(y_test, pred))

