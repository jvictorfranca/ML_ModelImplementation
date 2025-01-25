from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

ambev = pd.read_excel('Datasets/ambev.xlsx')
ambev
receita=pd.Series(ambev.iloc[:,2].values,
                  index=pd.date_range(start='2000-01-01', periods=len(ambev),
                                      freq='Q'))
bambev = receita[:-9]
reais = receita[-9:]

modelos = []
mapes = []

print(bambev)

print(reais)

# Modelo Naive
naive_forecast = pd.Series([bambev.iloc[-1]] * len(reais), index=reais.index)
print(naive_forecast)
mape_naive = mape(reais, naive_forecast)*100
modelos.append("Naive")
mapes.append(mape_naive)
print(mape_naive)

# Modelo Mean (média)
mean_forecast = pd.Series(bambev.mean(), index=reais.index)
print(mean_forecast)
mape_mean = mape(reais, mean_forecast)*100
modelos.append("Mean")
mapes.append(mape_mean)
print(mape_mean)

# Modelo Drift
n = len(bambev)
drift_slope = (bambev.iloc[-1] - bambev.iloc[0]) / (n - 1)
drift_forecast = bambev.iloc[-1] + drift_slope * np.arange(1, len(reais) + 1)
drift_forecast = pd.Series(drift_forecast, index=reais.index)
print(drift_forecast)
mape_drift = mape(reais, drift_forecast)*100
print(mape_drift)

# Modelo Naive Sazonal
naive_sazonal_forecast = pd.Series([bambev.iloc[-4 + (i % 4)]
                                    for i in range(len(reais))],
                                   index=reais.index)
print(naive_sazonal_forecast)
mape_naive_sazonal = mape(reais, naive_sazonal_forecast)*100
modelos.append("Naive Sazonal")
mapes.append(mape_naive_sazonal)
print(mape_naive_sazonal)

# Suavização Exponencial Simples (SES)
ses_model = SimpleExpSmoothing(bambev).fit()
ses_forecast = ses_model.forecast(steps=len(reais))
print(ses_forecast)
mape_ses = mape(reais, ses_forecast)*100
modelos.append("SES")
mapes.append(mape_ses)
print(mape_ses)

# Holt com Tendência
holt_model = Holt(bambev).fit()
holt_forecast = holt_model.forecast(steps=len(reais))
print(holt_forecast)
mape_holt = mape(reais, holt_forecast)*100
modelos.append("Holt")
mapes.append(mape_holt)
print(mape_holt)

# Holt-Winters Aditivo
hw_add_model = ExponentialSmoothing(bambev, seasonal_periods=4, trend='add', seasonal='add').fit()
hw_add_forecast = hw_add_model.forecast(steps=len(reais))
print(hw_add_forecast)
mape_hw_add = mape(reais, hw_add_forecast)*100
modelos.append("Holt-Winters Aditivo")
mapes.append(mape_hw_add)
print(mape_hw_add)

# Holt-Winters Multiplicativo
hw_mult_model = ExponentialSmoothing(bambev, seasonal_periods=4, trend='add', seasonal='mul').fit()
hw_mult_forecast = hw_mult_model.forecast(steps=len(reais))
print(hw_mult_forecast)
mape_hw_mult = mape(reais, hw_mult_forecast)*100
modelos.append("Holt-Winters Multiplicativo")
mapes.append(mape_hw_mult)
print(mape_hw_mult)

# Comparação dos modelos com base no MAPE
mape_comparison = pd.DataFrame({'Modelo': modelos, 'MAPE': mapes})
mape_comparison = mape_comparison.sort_values(by='MAPE', ascending=False).reset_index(drop=True)
print(mape_comparison)

plt.figure(figsize=(10, 6))
plt.barh(mape_comparison['Modelo'], mape_comparison['MAPE'], color='skyblue')
plt.xlabel("MAPE")
plt.title("MAPE Comparação de Modelos")
plt.grid(True)
plt.show()

# NOTE: Smaller MAPE is better