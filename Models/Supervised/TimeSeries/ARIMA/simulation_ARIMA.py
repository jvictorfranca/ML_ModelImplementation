import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

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

plt.figure(figsize=(10, 6))
plt.plot(serie_ar)
plt.title('Modelo AR(1) X(t)=0.8.X(t-1) + erro(t)')
plt.xlabel('Tempo')
plt.ylabel('Valores simulados')
plt.grid(True)
plt.show()


## Simulação de um modelo MA(1)
# Definir o coeficiente do modelo MA(1)

ma = np.array([1, -0.3])  # MA(1) com coeficiente -0.3
ar = np.array([1])  # Não há parte AR, então é apenas [1]

# Criar o processo MA(1)
ma_process = ArmaProcess(ar, ma)

# Simular 500 pontos para a série temporal
np.random.seed(42)  # Para reprodutibilidade
serie_ma = ma_process.generate_sample(nsample=500)

# Plotar a série temporal simulada
plt.figure(figsize=(10, 6))
plt.plot(serie_ma)
plt.title('Modelo MA(1) X(t)=-0.3erro(t-1) + erro(t)')
plt.xlabel('Tempo')
plt.ylabel('Valores simulados')
plt.grid(True)
plt.show()

## Simulação de um modelo ARMA(1,1)
# Definir os coeficientes do modelo ARMA(1,1)
ar = np.array([1, -0.8])  # AR(1) com coeficiente +0.8
ma = np.array([1, -0.3])  # MA(1) com coeficiente -0.3

# Criar o processo ARMA(1,1)
arma_process = ArmaProcess(ar, ma)

# Simular 500 pontos para a série temporal
np.random.seed(42)  # Para reprodutibilidade
serie_arma = arma_process.generate_sample(nsample=500)

# Plotar a série temporal simulada
plt.figure(figsize=(10, 6))
plt.plot(serie_arma)
plt.title('Simulação do Modelo ARMA(1,1) com AR=0.8 e MA=-0.3')
plt.xlabel('Tempo')
plt.ylabel('Valores simulados')
plt.grid(True)
plt.show()

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

#Visualizar a série simulada não estacionária
plt.figure(figsize=(10, 6))
plt.plot(serie_arima_nao_estacionaria)
plt.title("Série Não Estacionária ARIMA(1,1,1)")
plt.grid(True)
plt.show() 