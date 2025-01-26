# modelos ARIMA com Sazonalidade - SARIMA, possui os parâmetros P, D e Q Sazonais.
# Fica SARIMA(p,d,q)(P,D,Q)

# Buscando a série do Índice de Volume de Vendas de SP
# Pelo pacote python bcb - Baixar dados do Sistema Gerador de Séries Temporais
# do Banco Central

# https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries

# pip install python-bcb
from bcb import sgs
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import pandas as pd
from pmdarima import auto_arima

# Obter os dados da série do Índice de Volume de Vendas de SP do BCB
varejo2 = sgs.get({'volume_vendas': 1475}, start='2000-01-01', end='2022-12-31')
print(varejo2)

varejo2.index = pd.to_datetime(varejo2.index)
varejo2 = varejo2.asfreq('MS')
print(varejo2)

# Divisão da série em treino e teste
varejotreino = varejo2[:'2020-12']
varejoteste = varejo2['2021-01':]

# Checagem do tamanho do conjunto de teste
print(f"Comprimento da série de teste: {len(varejoteste)}")

result = adfuller(varejotreino.dropna())
print(f'Resultado do Teste ADF: p-valor = {result[1]}')
if result[1] < 0.05:
    print("A série é estacionária.")
else:
    print("A série não é estacionária.")

def verificar_differenciacao(serie, nome):
    # Usar a função ndiffs do pmdarima
    d = pm.arima.ndiffs(serie, test='adf')  # Teste de Dickey-Fuller aumentado
    print(f"A série {nome} precisa de {d} diferenciação(ões) para ser estacionária.")
    return d


#: Verificar quantas diferenciacoes sao necessarias
verificar_differenciacao(varejotreino, "Varejo - Treinamento")

# Diferenciação para estacionariedade
varejotreino_diff = varejotreino.diff().dropna()


arimavarejo = auto_arima(varejotreino_diff,
                         seasonal=True,
                         m=12,  # Periodicidade da sazonalidade
                         trace=True,
                         stepwise=True)

# Exibir o resumo do modelo ajustado
print(arimavarejo.summary())

#: Prever 24 passos à frente na série diferenciada
n_periods = 24
previsoes_diff = arimavarejo.predict(n_periods=n_periods)
print(f"Previsões diferenciadas: {previsoes_diff}")

# Índices das previsões (mesmo formato de data da série de treino e teste)
index_of_fc = pd.date_range(varejotreino.index[-1], periods=n_periods+1, freq='MS')[1:]

# Para voltar ao nível original:
# Iterar para reverter a diferenciação das previsões
ultimo_valor_original = varejotreino.iloc[-1] # Último valor conhecido da série original (não diferenciada)
previsoes_nivel_original = [ultimo_valor_original]
print(ultimo_valor_original)
print(previsoes_nivel_original)

#Somar as previsões diferenciadas ao último valor conhecido da série original
for previsao in previsoes_diff:
    novo_valor = previsoes_nivel_original[-1] + previsao
    previsoes_nivel_original.append(novo_valor)

#Remover o primeiro valor, pois é o último valor conhecido da série original
previsoes_nivel_original = previsoes_nivel_original[1:]
print(previsoes_nivel_original)

# Converter previsões de volta para uma Série Pandas com o índice correto
previsoes_nivel_original_series = pd.Series(previsoes_nivel_original, index=index_of_fc)
print(previsoes_nivel_original_series)

#  Garantir que as previsões e os valores reais estejam alinhados para o MAPE
previsoes_series_alinhadas = previsoes_nivel_original_series[:len(varejoteste)].dropna()
varejoteste_alinhada = varejoteste.loc[previsoes_series_alinhadas.index]

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(varejoteste_alinhada, previsoes_series_alinhadas)*100
print(f'MAPE: {mape}')
