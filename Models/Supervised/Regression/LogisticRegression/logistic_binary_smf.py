import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

df_atrasado = pd.read_csv('Datasets/atrasado.csv',delimiter=',')
print(df_atrasado)

modelo_atrasos = smf.glm(formula='atrasado ~ dist + sem', data=df_atrasado,
                         family=sm.families.Binomial()).fit()

print('logLike: {}'.format(modelo_atrasos.llf))

print(modelo_atrasos.summary())

print(modelo_atrasos.params)

#Making predictions
prediction = modelo_atrasos.predict(pd.DataFrame({'dist':[7], 'sem':[10]}))
print(prediction)
df_atrasado['phat'] = modelo_atrasos.predict()

print(df_atrasado)

#Calculating probability manually:

def calculating_probability(alpha, betas, parameters):
    # Making alpha + beta1*x1 + beta2*x2...
    beta_parameters_sum = 0
    for index, curr_beta in enumerate(betas):
        curr_parameter = parameters[index]
        beta_parameters_sum += curr_beta*curr_parameter

    # Calculating the probability
    probability = (1)/(1 + np.exp(-(alpha + beta_parameters_sum)))
    return probability

alpha = modelo_atrasos.params.iloc[0]
print(alpha)

betas = modelo_atrasos.params.iloc[1:]
print(betas)

prob = calculating_probability(alpha, betas, [7, 10])
print(prob)
