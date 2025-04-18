import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score

titanic = sns.load_dataset('titanic')

titanic['age'] = titanic.age.fillna(titanic.age.mean())
titanic.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 
                      'alive', 'alone'], inplace=True)

titanic_dummies = pd.get_dummies(titanic, drop_first=True)
X = titanic_dummies.drop(columns = ['survived'])
y = titanic_dummies['survived']

arvore = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state=42)

arvore.fit(X, y)

novos_dados = X.tail()
print(novos_dados)


classificação_novos_dados = arvore.predict(novos_dados)
classificação_novos_dados

# Store classification values
classificação_treino = arvore.predict(X)

# Makes manually the confusion matrix
print(pd.crosstab(classificação_treino, y, margins=True))
print(pd.crosstab(classificação_treino, y, normalize='index'))
print(pd.crosstab(classificação_treino, y, normalize='columns'))

# Creates the confusion matrix
cm = confusion_matrix(y, arvore.predict(X))

# accuracy_score calculation
ac = accuracy_score(y, arvore.predict(X))

# Force the target as a uniform distribution
bac = balanced_accuracy_score(y, arvore.predict(X))

# Heatmap with confusion matrix
sns.heatmap(cm, 
            annot=True, fmt='d', cmap='viridis', 
            xticklabels=['Não Sobreviveu', 'Sobreviveu'], 
            yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.show()

# Scikit learn classification report
print('\n', classification_report(y, arvore.predict(X)))