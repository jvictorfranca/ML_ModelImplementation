from scipy.stats import boxcox
import pandas as pd
import statsmodels.api as sm

df_bebes = pd.read_csv('Datasets/bebes.csv')

yast, lmbda = boxcox(df_bebes['comprimento'])

print('lambda: {f}'.format(f=lmbda))

df_bebes['bc_comprimento'] = yast

print(df_bebes)


#Verifying calculating the data manually
# ----------------------------
def transform_lambda_value(x, lmbda):
    return ((x**lmbda)-1)/lmbda

df_bebes['bc_comprimento2'] = transform_lambda_value(df_bebes['comprimento'], lmbda)

print(df_bebes)

del df_bebes['bc_comprimento2']

# ----------------------------



modelo_bc = sm.OLS.from_formula('bc_comprimento ~ idade', df_bebes).fit()
modelo_bc.summary()

r_squared = pd.DataFrame({'R² Box-Cox':[round(modelo_bc.rsquared,4)]})
print(r_squared)

# Inverse calculation with boxcox

def returm_value_from_lambda(x, lmbda):
    return ((x * lmbda) + 1) ** (1 / lmbda)

df_bebes['yhat_modelo_bc'] = returm_value_from_lambda(modelo_bc.fittedvalues, lmbda)
print(df_bebes)


# Making predictions 

prediction_bc = modelo_bc.predict(pd.DataFrame({'idade':[52]}))
print(prediction_bc)

predic_value = returm_value_from_lambda(prediction_bc, lmbda)
print(predic_value)