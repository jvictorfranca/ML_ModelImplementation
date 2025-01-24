import pingouin as pg
import pandas as pd

df_paises = pd.read_csv('Datasets/paises.csv', delimiter=',', encoding="utf-8")
df_paises

#Características das variáveis do dataset
df_paises.info()

#Estatísticas univariadas
df_paises.describe()

correlation_matrix = pg.rcorr(df_paises, method='pearson',
                               upper='pval', decimals=6,
                               pval_stars={0.01: '***',
                                           0.05: '**',
                                           0.10: '*'})
print(correlation_matrix)