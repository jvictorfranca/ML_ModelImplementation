from statsmodels.tsa.api import Holt
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


#Holt smoothing model with tendency

base = pd.Series([3, 5, 9, 20, 12, 17, 22, 23, 51, 41, 56, 75, 60, 75, 88])

holt_model = Holt(base).fit()
print(holt_model.summary())

# create forecast
holt_forecast = holt_model.forecast(3)


# See forecast
print("Previsão com Holt: ")
print(holt_forecast)

plt.plot(base, label="Dados Originais")
plt.plot(holt_model.fittedvalues, label="Ajustado (Holt)")
plt.plot(np.arange(len(base), len(base) + 3), holt_forecast, label="Previsão (Holt)")
plt.title("Modelo de Holt com Tendência")
plt.legend()
# plt.show()