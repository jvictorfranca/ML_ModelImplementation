# For residues auto-correlation
#Note: If H0 is rejected (as it is in the example), another category of modeling should be used, such as ARIMA

from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

energia = pd.read_excel("Datasets/energia.xlsx", usecols=[1]).dropna()
energia.index = pd.date_range(start='1979-01', periods=len(energia), freq='M')
energia = energia.squeeze()  # Converter para uma Series

# In[134]: Separar a base de dados em treino e teste

benergia = energia[:'2022-06'].ffill()  # Preencher valores nulos com forward fill
reaisenergia = energia['2022-07':'2024-06']  # Teste de 2022-07 até 2024-06

hw_mult_model = ExponentialSmoothing(
        benergia,
        seasonal_periods=12,
        trend='add',
        seasonal='mul',
        initialization_method="estimated",  # Método robusto de inicialização
        use_boxcox=True  # Tentar estabilizar a variância com Box-Cox
    ).fit(optimized=True)

hw_mul_forecast = hw_mult_model.forecast(steps=len(reaisenergia))


residuos = reaisenergia - hw_mul_forecast
print(residuos)

# In[150]: Teste de Ljung-Box para autocorrelação dos resíduos
lb_test = acorr_ljungbox(residuos, lags=[10], return_df=True)
print(f"Teste Ljung-Box:\n{lb_test}")

# Interpretação do teste de Ljung-Box
p_value_ljungbox = lb_test['lb_pvalue'].values[0]
if p_value_ljungbox > 0.05:
    print("Não há evidências de autocorrelação significativa nos resíduos (não rejeitamos H0).")
else:
    print("Há evidências de autocorrelação nos resíduos (rejeitamos H0).")
