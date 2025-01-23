import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats #chi2 statistic


## McFadden (Prêmio Nobel de Economia em 2000)

df_atrasado = pd.read_csv('Datasets/atrasado.csv',delimiter=',')
print(df_atrasado)

# Create a model without any betas
modelo_nulo = sm.Logit.from_formula('atrasado ~ 1',
                                        data=df_atrasado).fit()

modelo_atrasos = sm.Logit.from_formula('atrasado ~ dist + sem',
                                        data=df_atrasado).fit()

# Parâmetros do 'modelo_nulo'
print(modelo_nulo.summary())

# Loglike do 'modelo_nulo'
modelo_nulo.llf


pseudor2 = ((-2*modelo_nulo.llf)-(-2*modelo_atrasos.llf))/(-2*modelo_nulo.llf)
print(pseudor2)


# AIC (Akaike Info Criterion)
aic = -2*(modelo_atrasos.llf) + 2*(3)
print(aic)

modelo_atrasos.aic

# BIC (Bayesian Info Criterion)
bic = -2*(modelo_atrasos.llf) + 3*np.log(100)
print(bic)

bic_model = modelo_atrasos.bic
print(bic_model)
