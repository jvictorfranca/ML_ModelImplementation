import pandas as pd
import statsmodels.api as sm
from scipy import stats #chi2 statistic

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

chi2 = -2*(modelo_nulo.llf - modelo_atrasos.llf)
print(chi2)

# Freedon degrees = number of parameters in the final model
freedom_degrees = len(modelo_atrasos.params) - len(modelo_nulo.params) 
print(freedom_degrees)

pvalue = stats.distributions.chi2.sf(chi2, freedom_degrees)
print(pvalue)