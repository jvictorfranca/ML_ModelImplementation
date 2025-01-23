import pandas as pd
import statsmodels.api as sm
from statstests.process import stepwise

df_challenger = pd.read_csv('Datasets/challenger.csv',delimiter=',')
print(df_challenger)
print(df_challenger.describe())

#Creating a variable for failure
df_challenger.loc[df_challenger['desgaste'] != 0 , 'falha'] = 1
df_challenger.loc[df_challenger['desgaste'] == 0, 'falha'] = 0

#Give a p-value to reject H0 of 5% to validade the statisctical significance of the variables

modelo_challenger = sm.Logit.from_formula('falha ~ temperatura + pressão',
                                          df_challenger).fit()

step_challenger = stepwise(modelo_challenger, pvalue_limit=0.05)

print(step_challenger.summary())

prediction = step_challenger.predict(pd.DataFrame({'temperatura':[70]}))
print(prediction)

df_challenger['phat'] = step_challenger.predict()

print(df_challenger)