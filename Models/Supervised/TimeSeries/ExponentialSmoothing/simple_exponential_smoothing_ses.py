
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#Simple smoothing model without tendency or sasonality

base = pd.Series([3, 5, 9, 20, 12, 17, 22, 23, 51, 41, 56, 75, 60, 75, 88])

ses_model = SimpleExpSmoothing(base).fit()
print(ses_model.summary())

# Forecast

ses_forecast = ses_model.forecast(3)

print("Previsões (SES):")
print(ses_forecast)

# Valores ajustados

print("Valores Ajustados (SES):")
print(ses_model.fittedvalues)

plt.plot(base, label="Base de Dados")
plt.plot(ses_model.fittedvalues, label="Ajustado (SES)")
plt.plot(np.arange(len(base), len(base) + 3), ses_forecast, label="Previsão (SES)")
plt.title("Suavização Exponencial Simples (SES)")
plt.legend()
# plt.show()