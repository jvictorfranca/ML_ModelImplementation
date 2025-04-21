
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
    StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

titanic = pd.read_pickle('Datasets/pickle/titanic.pkl')
X = titanic.drop(columns='survived')
y=titanic.survived

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

param_dist = {
    'n_estimators': [50, 100, 200, 300],  # Número de árvores
    'max_depth': range(2, 30, 1),      # Profundidade máxima das árvores
    'min_samples_split': [2, 5, 10],      # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],        # Número mínimo de amostras em uma folha
    'max_features': ['sqrt', 'log2', None],  # Número de features consideradas para divisão
    'bootstrap': [True, False],           # Usar bootstrap ou não
    'criterion': ['gini', 'entropy'],     # Critério de divisão
    'ccp_alpha': np.linspace(0, 0.05, 20)  # Parâmetro de poda de complexidade de custo
}

n_iter = 50  # Número de combinações de hiperparâmetros a serem testadas
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold estratificado
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=kf,
    scoring='roc_auc',
    n_jobs=-1,  # Usar todos os núcleos do processador
    verbose=1,  # Mostrar progresso
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"\nMelhores hiperparâmetros: {random_search.best_params_}")
print(f"\nAUC média na validação cruzada: {random_search.best_score_:.4f}")

final_clf = random_search.best_estimator_

random_test_score = final_clf.score(X_test, y_test)
random_roc = roc_auc_score(y_test, final_clf.predict_proba(X_test)[:,1])
random_gini = random_roc*2-1

resultados = pd.DataFrame(random_search.cv_results_)
resultados['gini'] = resultados.mean_test_score*2-1