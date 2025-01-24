# To test normalities of the residues

# Teste de Shapiro-Wilk (n < 30)
# from scipy.stats import shapiro
# shapiro(modelo_linear.resid)

from scipy.stats import shapiro
import pandas as pd
import statsmodels.api as sm

df_bebes = pd.read_csv('Datasets/bebes.csv')
print(df_bebes)


modelo_linear = sm.OLS.from_formula('comprimento ~ idade', df_bebes).fit()

# Parâmetros do 'modelo_linear'
modelo_linear.summary()

## Teste de Shapiro-Wilk: interpretação

# criação do objeto 'teste_sf'
teste_sf = shapiro(modelo_linear.resid)

# retorna o grupo de pares de valores-chave no dicionário


# definição dos elementos da lista (tupla)
statistics_z = teste_sf.statistic
p = teste_sf.pvalue 

print('Statistics Z=%.5f, p-value=%.6f' % (statistics_z ,p))

# Definição do nível de significância
alpha = 0.05
if p > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')