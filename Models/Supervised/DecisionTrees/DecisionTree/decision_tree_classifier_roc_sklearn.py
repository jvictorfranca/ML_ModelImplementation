
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def avalia_clf(clf, y, X, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'treino'):
    
    # Calcular as classificações preditas
    pred = clf.predict(X)
    
    # Calcular a probabilidade de evento
    y_prob = clf.predict_proba(X)[:, -1]
    
    # Calculando acurácia e matriz de confusão
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'A acurácia da árvore é: {ac:.1%}')
    print(f'A acurácia balanceada da árvore é: {bac:.1%}')
    
    # Calculando AUC
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    print(f"GINI: {(2*auc_score-1):.2%}")
    
    # Visualização gráfica
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='viridis', 
                xticklabels=rótulos_y, 
                yticklabels=rótulos_y)
    
    # Relatório de classificação do Scikit
    print('\n', classification_report(y, pred))
    
    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


titanic = sns.load_dataset('titanic')

titanic['age'] = titanic.age.fillna(titanic.age.mean())
titanic.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 
                      'alive', 'alone'], inplace=True)


titanic_dummies = pd.get_dummies(titanic, drop_first=True)
X = titanic_dummies.drop(columns = ['survived'])
y = titanic_dummies['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)


arvore = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state=42)

arvore.fit(X, y)

avalia_clf(arvore, y_train,X_train)

avalia_clf(arvore, y_test,X_test)