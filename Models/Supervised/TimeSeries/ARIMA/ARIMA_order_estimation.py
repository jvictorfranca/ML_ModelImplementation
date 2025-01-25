import pmdarima as pm
from simulation_ARIMA import serie_ar, serie_ma, serie_arma, serie_arima_nao_estacionaria
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def simular_arima(n, ar=[1, -0.8], ma=[1, -0.3], d=1, noise_std=5):
    """Simula uma série ARIMA com tendência, maior variabilidade e componente não sazonal."""
    
    # Criar a parte ARMA (ARIMA sem diferenciação)
    ar_params = np.r_[1, -np.array(ar[1:])]  # Parâmetro AR ajustado para ARMAProcess
    ma_params = np.r_[1, np.array(ma[1:])]   # Parâmetro MA ajustado para ARMAProcess
    arma_process = ArmaProcess(ar_params, ma_params)
    serie_arma = arma_process.generate_sample(nsample=n)
    
    # Adicionar um componente de tendência (para garantir que a série seja não estacionária)
    tendencia = np.linspace(0, n * 0.05, n)  # Componente de tendência linear
    
    # Adicionar variabilidade adicional
    variabilidade_adicional = np.random.normal(loc=0, scale=noise_std, size=n)  # Variabilidade adicional
    
    # Adicionar a tendência, variabilidade e aplicar a diferenciação (d=1)
    serie_arima = np.cumsum(serie_arma + tendencia + variabilidade_adicional)  # Diferenciação inversa (integração)
    
    return pd.Series(serie_arima)

# Simular a série ARIMA(1,1,1) com tendência e variabilidade aumentada
np.random.seed(42)
serie_arima = simular_arima(500, ar=[1, -0.8], ma=[1, -0.3], d=1, noise_std=5)

# Função para verificar quantas diferenciações são necessárias para tornar a série estacionária

def verificar_differenciacao(serie, nome):
    # Usar a função ndiffs do pmdarima
    d = pm.arima.ndiffs(serie, test='adf')  # Teste de Dickey-Fuller aumentado
    print(f"A série {nome} precisa de {d} diferenciação(ões) para ser estacionária.")
    return d

# Verificar quantas diferenciações são necessárias
verificar_differenciacao(serie_arima, "ARIMA(1,1,1)")

# Verificar quantas diferenciações são necessárias
verificar_differenciacao(serie_ar, "AR(1)")
verificar_differenciacao(serie_ma, "MA(1)")
verificar_differenciacao(serie_arma, "ARMA(1,1)")


# Simulação de séries AR de ordens maiores

# Função para simular séries AR, MA e ARIMA
def simular_arima(ar=None, ma=None, n=500, d=1, seed=42):
    np.random.seed(seed)
    ar = np.array([1] + [-coef for coef in (ar if ar else [])])  # Definir AR
    ma = np.array([1] + [coef for coef in (ma if ma else [])])  # Definir MA
    process = ArmaProcess(ar, ma)
    return process.generate_sample(nsample=n)

serie_ar2 = simular_arima(ar=[0.8, 0.1], n=500)

serie_ar3 = simular_arima(ar=[0.5, 0.1, 0.3], n=500)

def teste_dickey_fuller(serie, nome):
    resultado = adfuller(serie)
    print(f"\nTeste Dickey-Fuller para {nome}:")
    print(f"Estatística: {resultado[0]}")
    print(f"P-valor: {resultado[1]}")
    for key, value in resultado[4].items():
        print(f"{key}: {value}")
    print('Conclusão:', 'Estacionária' if resultado[1] < 0.01 else 'Não Estacionária')

# Verificar a estacionariedade das séries simuladas
teste_dickey_fuller(serie_ar2, "AR(2)")

teste_dickey_fuller(serie_ar3, "AR(3)")

# Identificar modelos ARIMA diretamente

## Estimar ARIMA manualmente com ordem (2,0,0) para série 2 e (3,0,0) para série 3
modelo_ar2 = ARIMA(serie_ar2, order=(2, 0, 0)).fit()
print(f'\nModelo ARIMA(2,0,0) ajustado para serie2:\n{modelo_ar2.summary()}')

modelo_ar3 = ARIMA(serie_ar3, order=(3, 0, 0)).fit()
print(f'\nModelo ARIMA(3,0,0) ajustado para serie3:\n{modelo_ar3.summary()}')

# Simular séries AR(3) com diferentes coeficientes positivos e negativos
## AR(3) com coeficientes [0.5, -0.1, -0.3]
serie_ar31 = simular_arima(ar=[0.5, -0.1, -0.3], n=500)

# Plot ACF e PACF para serie2 (AR(3) com coeficientes [0.5, 0.1, 0.3])

modelo_ar31 = ARIMA(serie_ar31, order=(3, 0, 0)).fit()
print(f'\nModelo ARIMA(3,0,0) ajustado para AR(3) com coef positivos e negativos:\n{modelo_ar31.summary()}')


## Simular uma série ARMA(2,2)
serie_arma221 = simular_arima(ar=[0.8, -0.1], ma=[0.4, -0.3], n=500)

# Testar a estacionariedade de série ARMA(2,2)
teste_dickey_fuller(serie_arma221, "ARMA(2,2) coef positivos e negativos")

### Caso encontre ARIMA(0,0,0) - não foi possível encontrar memória
## autoregressiva significativa