#Acá es donde voy a hacer el barrido eligiendo los parámetros con el lhs
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_1 as gucd
import latin_hypercube_sampling as lhs
from time import time
import pandas as pd
plt.ion()

#%%
#El análisis
df = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb'])

n_parametros = 8
n_barrido = 10
parametros = lhs.lhs(n_parametros, n_barrido)

areas = []

ti = time()
for params in parametros:
    A_s, B_s, S = gucd.gucd_modelo_1(*params, 1000, 2, 50)
    area = mb.mide_biestabilidad(A_s, S)
    if area != None:
        if area > 0:
            df1 = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb'], index=[str(area)])
            df1.loc[str(area), :] = params
            df = df.append(df1)
            del(df1)
    del(A_s, B_s, S, area)
tf = time()
print(f'Tarda {tf -ti} segundos en hacer todo el análisis para 10 sets de parámetros.')
df.to_csv('2021_11_29-parametros_biestables.csv')
