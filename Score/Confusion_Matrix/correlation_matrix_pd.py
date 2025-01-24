import pandas as pd

df_paises = pd.read_csv('Datasets/paises.csv', delimiter=',', encoding="utf-8")
df_paises

#Características das variáveis do dataset
df_paises.info()

#Estatísticas univariadas
df_paises.describe()

correlation_matrix = df_paises.iloc[:,1:4].corr()
print(correlation_matrix)