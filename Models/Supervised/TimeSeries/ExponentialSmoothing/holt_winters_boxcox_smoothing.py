from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error as mape
import pandas as pd

energia = pd.read_excel("Datasets/energia.xlsx", usecols=[1]).dropna()
energia.index = pd.date_range(start='1979-01', periods=len(energia), freq='M')
energia = energia.squeeze()  # Converter para uma Series

# In[134]: Separar a base de dados em treino e teste

benergia = energia[:'2022-06'].ffill()  # Preencher valores nulos com forward fill
reaisenergia = energia['2022-07':'2024-06']  # Teste de 2022-07 até 2024-06


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
    print(hw_add_forecast)

    mape_hw_add = mape(reaisenergia, hw_add_forecast) * 100
    print(mape_hw_add)
except Exception:
    print("error")

