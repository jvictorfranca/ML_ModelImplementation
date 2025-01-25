
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW

## Holt-Winters smoothing model with only tendency

base = pd.Series([3, 5, 9, 20, 12, 17, 22, 23, 51, 41, 56, 75, 60, 75, 88])

holt_winters_tendency = HW(base, trend='add', seasonal=None).fit()
print(holt_winters_tendency.summary())

fitted_holt_winters_tendency = holt_winters_tendency.fittedvalues
print("Valores ajustados (Holt-Winters com tendência):")
print(fitted_holt_winters_tendency)

# Previsão de 5 passos à frente
prevholt_winters_tendency = holt_winters_tendency.forecast(5)
print("Previsão para os próximos 5 períodos:")
print(prevholt_winters_tendency)

# In[88]: Visualização dos dados ajustados e previsão
plt.plot(base, label="Dados originais")
plt.plot(fitted_holt_winters_tendency, label="Ajustado (Holt-Winters com tendência)")
plt.plot(np.arange(len(base), len(base) + 5), prevholt_winters_tendency, label="Previsão")
plt.title("Holt-Winters com Tendência")
plt.legend()
# plt.show()


## Holt-Winters smoothing model with tendency and sasonality

base_sasonality = pd.Series([10, 14, 8, 25, 16, 22, 14, 35, 15, 27, 18, 40, 28, 40, 25, 65],
                  index=pd.date_range(start='2019-01-01', periods=16, freq='Q'))


def plot_holtwinters(model, fitted_values, forecast, model_type):
    plt.figure(figsize=(10, 6))

    # Obter datas para o período da previsão
    forecast_index = pd.date_range(start=base_sasonality.index[-1] + pd.offsets.QuarterEnd(), periods=len(forecast), freq='Q')

    # Plotando os dados originais, ajustados e previsão
    plt.plot(base_sasonality.index, base_sasonality, label="Dados Originais", marker='o', color='blue')
    plt.plot(base_sasonality.index, fitted_values, label="Valores Ajustados", marker='o', color='green')
    plt.plot(forecast_index, forecast, label="Previsão", marker='o', color='red')

    # Intervalo de confiança
    plt.fill_between(forecast_index, forecast * 0.95, forecast * 1.05, color='gray', alpha=0.2, label="Intervalo de Confiança 95%")

    plt.title(f"Modelo Holt-Winters ({model_type})", fontsize=14)
    plt.xlabel("Período", fontsize=12)
    plt.ylabel("Valores", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Create the model with seasonality

holt_winters_sasonality = HW(base_sasonality, trend='add', seasonal='add', seasonal_periods=4).fit()
fitted_holt_winters_sasonality = holt_winters_sasonality.fittedvalues
print(holt_winters_sasonality.summary())

prevholt_winters_sasonality = holt_winters_sasonality.forecast(4)
print("Previsão para os próximos 4 períodos:")
print(prevholt_winters_sasonality)

plot_holtwinters(fitted_holt_winters_sasonality, fitted_holt_winters_sasonality, prevholt_winters_sasonality, model_type="Aditivo")


## Holt-winters model with multiplicative seasonality

holt_winters_sasonality_mult = HW(base_sasonality, trend='add', seasonal='mul', seasonal_periods=4).fit()
fitted_holt_winters_sasonality_mult = holt_winters_sasonality_mult.fittedvalues
print(holt_winters_sasonality_mult.summary())

prevholt_winters_sasonality_mult = holt_winters_sasonality_mult.forecast(4)
prevholt_winters_sasonality_mult
plot_holtwinters(holt_winters_sasonality_mult, fitted_holt_winters_sasonality_mult, prevholt_winters_sasonality_mult, model_type="Multiplicativo")
