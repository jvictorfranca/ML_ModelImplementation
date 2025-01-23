import pandas as pd
import statsmodels.api as sm

df_challenger = pd.read_csv('Datasets/challenger.csv',delimiter=',')
print(df_challenger)
print(df_challenger.describe())

#Creating a variable for failure
df_challenger.loc[df_challenger['desgaste'] != 0 , 'falha'] = 1
df_challenger.loc[df_challenger['desgaste'] == 0, 'falha'] = 0

modelo_challenger = sm.Logit.from_formula('falha ~ temperatura + pressão',
                                          df_challenger).fit()

print(modelo_challenger.summary())

prediction = modelo_challenger.predict(pd.DataFrame({'temperatura':[70], 'pressão': [60]}))
print(prediction)