import pandas as pd
import numpy as np

aleat = pd.Series(np.random.normal(size=500))

# Mean and standard error 
print("Media:", aleat.mean())
print("Desvio padrao:", aleat.std())

#Cumulative sum
passeio = aleat.cumsum()

