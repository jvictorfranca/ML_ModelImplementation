# To check if the time serie is stationary or not
# H0: A série Não é Estacionária
# H1: A série é Estacionária

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess

# Teste de Dickey-Fuller aumentado (ADF)
def dickey_fuller_test(series, title=''):
    result = adfuller(series)
    print(f'Teste de Dickey-Fuller para {title}')
    print(f'Estatística: {result[0]}')
    print(f'p-valor: {result[1]}')
    print('Critérios:')
    for key, value in result[4].items():
        print(f'{key}: {value}')
    print('Conclusão:', 'Estacionária' if result[1] < 0.01 else 'Não Estacionária')
    print()



## Simulação de um modelo AR(1)
# Definir o coeficiente do modelo AR(1)

ar = np.array([1, -0.8])

# AR(1) com coeficiente +0.8 (note o sinal negativo para simulação) a biblioteca ArmaProcess espera que o sinal seja inverso
ma = np.array([1])  # Não há parte MA, então é apenas [1]

# Criar o processo AR(1)
ar_process = ArmaProcess(ar, ma)

# Simular 500 pontos para a série temporal
np.random.seed(42)  # Para reprodutibilidade
serie_ar = ar_process.generate_sample(nsample=500)

## Simulação de um modelo MA(1)
# Definir o coeficiente do modelo MA(1)

ma = np.array([1, -0.3])  # MA(1) com coeficiente -0.3
ar = np.array([1])  # Não há parte AR, então é apenas [1]

# Criar o processo MA(1)
ma_process = ArmaProcess(ar, ma)

# Simular 500 pontos para a série temporal
np.random.seed(42)  # Para reprodutibilidade
serie_ma = ma_process.generate_sample(nsample=500)


## Simulação de um modelo ARMA(1,1)
# Definir os coeficientes do modelo ARMA(1,1)
ar = np.array([1, -0.8])  # AR(1) com coeficiente +0.8
ma = np.array([1, -0.3])  # MA(1) com coeficiente -0.3

# Criar o processo ARMA(1,1)
arma_process = ArmaProcess(ar, ma)

# Simular 500 pontos para a série temporal
np.random.seed(42)  # Para reprodutibilidade
serie_arma = arma_process.generate_sample(nsample=500)

## Simulando um modelo ARIMA(1,1,1)

# Definir o número de pontos para simulação
pontos = 500

# Definir os parâmetros ARIMA (1,1,1)
ar = np.array([1, -0.8])   
ma = np.array([1, -0.3])   
 
# Simular a série temporal ARIMA(1,1,1)
np.random.seed(42)  # Para reprodutibilidade
arma_process = ArmaProcess(ar, ma)
serie_arima = arma_process.generate_sample(nsample=pontos)

# Converter a série estacionária em uma série não estacionária aplicando a integração (d=1)
serie_arima_nao_estacionaria = np.cumsum(serie_arma)  # Diferenciação inversa (integração)

# Converter a série simulada em um DataFrame
serie_arima_nao_estacionaria = pd.Series(serie_arima_nao_estacionaria)

# Aplicando o teste de Dickey-Fuller

dickey_fuller_test(serie_ar, 'AR(1)')

dickey_fuller_test(serie_ma, 'MA(1)')

dickey_fuller_test(serie_arma, 'ARMA(1,1)')

# ATENÇÃO: vamos rodar para a série ARIMA, tem o I = 1
dickey_fuller_test(serie_arima_nao_estacionaria, 'ARIMA(1,1,1)')