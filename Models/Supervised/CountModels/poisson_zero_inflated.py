# LAMBERT, D. Zero-inflated Poisson regression, with an application to defects
#in manufacturing. Technometrics, v. 34, n. 1, p. 1-14, 1992.

# Exemplo de uma função da distribuição ZI Poisson, com lambda=1 e plogit=0,7

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import numpy as np

#'statsmodels.discrete.count_model'

# Definição da variável dependente (voltando ao dataset 'df_corruption')

df_corruption = pd.read_csv('Datasets/corruption.csv', delimiter=',')
print(df_corruption)

y = df_corruption['violations']

# Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_corruption[['staff','post','corruption']]
X1 = sm.add_constant(x1)

# Dummização da variável 'post'
# Se estimarmos o modelo sem dummizar as variáveis categóricas, o modelo retorna
#um erro
X1 = pd.get_dummies(X1, columns=['post'], dtype=int, drop_first=True)


# Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_corruption[['corruption']]
X2 = sm.add_constant(x2)

# O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zip = sm.ZeroInflatedPoisson(y, X1, exog_infl=X2,
                                    inflation='logit').fit()

# Parâmetros do 'modelo_zip'
modelo_zip.summary()