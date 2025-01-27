# Estimação do modelo ZINB pela função 'ZeroInflatedNegativeBinomialP' do pacote
#'statsmodels.discrete.count_model'

from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import pandas as pd
import statsmodels.api as sm

df_corruption = pd.read_csv('Datasets/corruption.csv', delimiter=',')

# Definição da variável dependente (voltando ao dataset 'df_corruption')
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
modelo_zinb = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2,
                                            inflation='logit').fit()

# Parâmetros do 'modelo_zinb'
modelo_zinb.summary()


# O parâmetro 'alpha' representa o 'fi' e é o inverso do parâmetro 'theta',
#ou seja, o inverso do parâmetro de forma da distribuição Poisson-Gama.
# Como 'alpha' (e da mesma forma 'theta') é estatisticamente diferente de
#zero, podemos afirmar que há superdispersão nos dados (outra forma de
#verificar o fenômeno da superdispersão!)
