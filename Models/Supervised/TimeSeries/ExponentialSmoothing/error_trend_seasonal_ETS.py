# Gets the best model, using different parameters, such as tendecy additive, multiplicative, seasonality additive, multiplicative, etc.

from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
 
# Importando os dados da série temporal da Ambev
ambev = pd.read_excel('Datasets/ambev.xlsx')
ambev.head()

receita=pd.Series(ambev.iloc[:,2].values,
                  index=pd.date_range(start='2000-01-01', periods=len(ambev),
                                      freq='Q'))

# Separar a base de dados em treino e teste (janela de dados)
bambev = receita[:-9]
reais = receita[-9:]

# Definir todas as combinações possíveis de modelos para ETS
configs = [
    {'trend': None, 'seasonal': None},
    {'trend': 'add', 'seasonal': None},
    {'trend': None, 'seasonal': 'add'},
    {'trend': 'add', 'seasonal': 'add'}
]

best_aic = float('inf')
best_config = None
best_model = None

# Ajustar os modelos com diferentes configurações e comparar AIC
for config in configs:
    try:
        model = ExponentialSmoothing(bambev, seasonal_periods=4, trend=config['trend'], seasonal=config['seasonal']).fit()
        aic = model.aic

        if aic < best_aic:
            best_aic = aic
            best_config = config
            best_model = model
    except Exception as e:
        pass  # Ignorar configurações que não funcionam

print(f"Melhor configuração: {best_config} com AIC = {best_aic}")
print(best_model.summary())

best_forecasts = best_model.forecast(steps=9)
print("Previsão para os próximos 9 períodos:")
print(best_forecasts)
