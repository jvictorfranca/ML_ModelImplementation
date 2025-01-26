# to test for residue heterocedasticityfrom arch import arch_model

from arch import arch_model
from bcb import sgs
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from pmdarima import auto_arima

# Obter os dados da série do Índice de Volume de Vendas de SP do BCB
varejo2 = sgs.get({'volume_vendas': 1475}, start='2000-01-01', end='2022-12-31')
#print(varejo2)

varejo2.index = pd.to_datetime(varejo2.index)
varejo2 = varejo2.asfreq('MS')
#print(varejo2)

# Divisão da série em treino e teste
varejotreino = varejo2[:'2020-12']
varejoteste = varejo2['2021-01':]


varejotreino_diff = varejotreino.diff().dropna()


arimavarejo = auto_arima(varejotreino_diff,
                         seasonal=True,
                         m=12,  # Periodicidade da sazonalidade
                         trace=True,
                         stepwise=True)

residuos_arima = arimavarejo.resid()

am = arch_model(residuos_arima, vol='ARCH', p=1)
test_arch = am.fit(disp='off')
print(test_arch.summary())


#se p-value > 0.05 - nao ha efeitos ARCH