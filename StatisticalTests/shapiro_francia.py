# To test normalities of the residues

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.tests import shapiro_francia
import pandas as pd
import statsmodels.api as sm

df_bebes = pd.read_csv('Datasets/bebes.csv')
print(df_bebes)


modelo_linear = sm.OLS.from_formula('comprimento ~ idade', df_bebes).fit()

# Parâmetros do 'modelo_linear'
modelo_linear.summary()

## Teste de Shapiro-Francia: interpretação

# criação do objeto 'teste_sf'
teste_sf = shapiro_francia(modelo_linear.resid)

# retorna o grupo de pares de valores-chave no dicionário
teste_sf = teste_sf.items()
print(teste_sf)

# definição dos elementos da lista (tupla)
method, statistics_W, statistics_z, p = teste_sf 

print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))

# Definição do nível de significância
alpha = 0.05
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')