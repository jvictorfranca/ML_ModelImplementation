### Estimação de um modelo ARIMA - Escolher, p, q e d

from pmdarima import auto_arima
from simulation_ARIMA import serie_ar, serie_ma, serie_arma, serie_arima_nao_estacionaria
import numpy as np
import pandas as pd



# Estimar automaticamente o modelo ARIMA
# Lembrando: simulamos um AR(1) de coeficiente 0.8
auto_arima_model = auto_arima(serie_ar, trace=True, seasonal=False, stepwise=True)
print(auto_arima_model.summary())

# Lembrando: simulamos um MA(1) de coeficiente -0.3
auto_arima_model_ma = auto_arima(serie_ma, trace=True, seasonal=False, stepwise=True)
print(auto_arima_model_ma.summary())

# Lembrando: simulamos um ARMA(1,1) de coeficiente AR = 0.8 e MA= -0.3
auto_arima_model_arma = auto_arima(serie_arma, trace=True, seasonal=False, stepwise=True)
print(auto_arima_model_arma.summary())

# Lembrando: simulamos um ARIMA(1,1,1) de coeficiente AR = 0.8 e MA= -0.3
auto_arima_model_arima = auto_arima(serie_arima_nao_estacionaria, trace=True, seasonal=False, stepwise=True)

print(auto_arima_model_arima.summary())