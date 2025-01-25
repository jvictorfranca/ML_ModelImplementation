# Gets the best model, using different parameters, such as tendecy additive, multiplicative, seasonality additive, multiplicative, etc.

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
import warnings
# Capturar avisos de convergência e tratá-los
warnings.filterwarnings("error", category=ConvergenceWarning)


#Carregar os dados de energia e garantir que não há valores ausentes
energia = pd.read_excel("Datasets/energia.xlsx", usecols=[1]).dropna()

energia.index = pd.date_range(start='1979-01', periods=len(energia), freq='M')
energia = energia.squeeze()  # Converter para uma Series

#Separar a base de dados em treino e teste
benergia = energia[:'2022-06'].ffill()  # Preencher valores nulos com forward fill
reaisenergia = energia['2022-07':'2024-06']  # Teste de 2022-07 até 2024-06

benergia = pd.to_numeric(benergia, errors='coerce').astype(float)

# Lista para armazenar os modelos, MAPE e previsões
modelos_energia = []
mapes_energia = []
previsoes_energia = {}

## Modelo Naive
naive_forecast = pd.Series([benergia.iloc[-1]] * len(reaisenergia), index=reaisenergia.index)
mape_naive = mape(reaisenergia, naive_forecast) * 100
modelos_energia.append("Naive")
mapes_energia.append(mape_naive)
previsoes_energia["Naive"] = naive_forecast

## Modelo Mean (média)
mean_forecast = pd.Series(benergia.mean(), index=reaisenergia.index)
mape_mean = mape(reaisenergia, mean_forecast) * 100
modelos_energia.append("Mean")
mapes_energia.append(mape_mean)
previsoes_energia["Mean"] = mean_forecast

## Modelo Drift
n = len(benergia)
drift_slope = (benergia.iloc[-1] - benergia.iloc[0]) / (n - 1)
drift_forecast = benergia.iloc[-1] + drift_slope * np.arange(1, len(reaisenergia) + 1)
drift_forecast = pd.Series(drift_forecast, index=reaisenergia.index)
mape_drift_result = mape(reaisenergia, drift_forecast) * 100
modelos_energia.append("Drift")
mapes_energia.append(mape_drift_result)
previsoes_energia["Drift"] = drift_forecast

# Modelo Naive Sazonal
naive_sazonal_forecast = pd.Series([benergia.iloc[-12 + (i % 12)]
                                    for i in range(len(reaisenergia))],
                                   index=reaisenergia.index)
mape_naive_sazonal = mape(reaisenergia, naive_sazonal_forecast) * 100
modelos_energia.append("Naive Sazonal")
mapes_energia.append(mape_naive_sazonal)
previsoes_energia["Naive Sazonal"] = naive_sazonal_forecast

## Suavização Exponencial Simples (SES)
ses_model = SimpleExpSmoothing(benergia).fit(optimized=True)
ses_forecast = ses_model.forecast(steps=len(reaisenergia))
mape_ses = mape(reaisenergia, ses_forecast) * 100
modelos_energia.append("SES")
mapes_energia.append(mape_ses)
previsoes_energia["SES"] = ses_forecast

## Holt-Winters Aditivo - Ajustar inicialização e Box-Cox para melhorar a convergência
try:
    hw_add_model = ExponentialSmoothing(
        benergia,
        seasonal_periods=12,
        trend='add',
        seasonal='add',
        initialization_method="estimated",  # Método robusto de inicialização
        use_boxcox=True  # Tentar estabilizar a variância com Box-Cox
    ).fit(optimized=True)
    
    hw_add_forecast = hw_add_model.forecast(steps=len(reaisenergia))
    mape_hw_add = mape(reaisenergia, hw_add_forecast) * 100
    modelos_energia.append("Holt-Winters Aditivo")
    mapes_energia.append(mape_hw_add)
    previsoes_energia["Holt-Winters Aditivo"] = hw_add_forecast
except Exception:
    modelos_energia.append("Holt-Winters Aditivo")
    mapes_energia.append(np.nan)


## Holt-Winters Multiplicativo - Ajustar inicialização e Box-Cox para melhorar a convergência
try:
    hw_mult_model = ExponentialSmoothing(
        benergia,
        seasonal_periods=12,
        trend='add',
        seasonal='mul',
        initialization_method="estimated",  # Método robusto de inicialização
        use_boxcox=True  # Tentar estabilizar a variância com Box-Cox
    ).fit(optimized=True)
    
    hw_mult_forecast = hw_mult_model.forecast(steps=len(reaisenergia))
    mape_hw_mult = mape(reaisenergia, hw_mult_forecast) * 100
    modelos_energia.append("Holt-Winters Multiplicativo")
    mapes_energia.append(mape_hw_mult)
    previsoes_energia["Holt-Winters Multiplicativo"] = hw_mult_forecast
except Exception:
    modelos_energia.append("Holt-Winters Multiplicativo")
    mapes_energia.append(np.nan)

# Comparação dos modelos com base no MAPE
mape_comparison = pd.DataFrame({'Modelo': modelos_energia, 'MAPE': mapes_energia})
mape_comparison = mape_comparison.sort_values(by='MAPE', ascending=True).reset_index(drop=True)
print(mape_comparison)