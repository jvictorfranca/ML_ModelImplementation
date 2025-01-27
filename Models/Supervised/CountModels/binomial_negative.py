# In[11]: Estimação do modelo binomial negativo do tipo NB2

# O argumento 'family=sm.families.NegativeBinomial(alpha=2.0963)' da função
#'smf.glm' define a estimação de um modelo binomial negativo do tipo NB2
#com valor de 'fi' ('alpha' no Python) igual a 2.0963 (valor proveniente da
#estimação realizada por meio do Solver do Excel). Lembramos que 'fi' é o
#inverso do parâmetro de forma 'theta' da distribuição Poisson-Gama.

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import numpy as np


df_corruption = pd.read_csv('Datasets/corruption.csv', delimiter=',')
print(df_corruption)

# Características das variáveis do dataset
df_corruption.info()

# Estatísticas univariadas
df_corruption.describe()

modelo_bneg = smf.glm(formula='violations ~ staff + post + corruption',
                      data=df_corruption,
                      family=sm.families.NegativeBinomial(alpha=2.0963)).fit()

# Parâmetros do 'modelo_bneg'
modelo_bneg.summary()

# Construção de uma função para a definição do 'fi' ótimo (argumento 'alpha') que gera a maximização do valor de Log-Likelihood

# Tempo aproximado de estimação desta célula: 1 min 40 seg

from tqdm import tqdm # para mostrar o progresso do loop

n_samples = 10000
alphas = np.linspace(0, 10, n_samples)
llf = np.full(n_samples, fill_value=np.nan)

for i, alpha in tqdm(enumerate(alphas), total=n_samples, desc='Estimating'):
    try:
        model = smf.glm(formula='violations ~ staff + post + corruption',
                        data=df_corruption,
                        family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except:
        continue
    llf[i] = model.llf

fi_ótimo = alphas[np.nanargmax(llf)].round(4)
fi_ótimo

# Reestimação do modelo binomial negativo com o parâmetro 'fi_ótimo'

modelo_bneg = smf.glm(formula='violations ~ staff + post + corruption',
                      data=df_corruption,
                      family=sm.families.NegativeBinomial(alpha=fi_ótimo)).fit()

# Parâmetros do 'modelo_bneg'
modelo_bneg.summary()

modelo_bneg.predict(pd.DataFrame({'staff':[23],
                                  'post':['no'],
                                  'corruption':[0.5]}))
