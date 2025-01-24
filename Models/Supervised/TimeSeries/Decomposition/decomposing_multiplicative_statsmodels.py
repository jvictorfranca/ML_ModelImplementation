import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt



pib = pd.read_excel("Datasets/pib_mensal.xlsx", parse_dates=True, index_col=0)
 
# Transformar a base de dados em um objeto de classe ts
pib_ts = pd.Series(pib['pib'].values, index=pib.index)
 

decomp_mul = seasonal_decompose(pib_ts, model='multiplicative', period=12)
decomp_mul.plot()

plt.suptitle('Decomposicao Multiplicativa do PIB')
plt.show()


