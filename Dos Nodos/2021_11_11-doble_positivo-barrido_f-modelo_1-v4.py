#Acá tomo cosas del v1 para ver cómo usar todo esto para hacer el barrido con lhs
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
#Pruebo la función que hace el going up comming down para el modelo 1
A_s, B_s, S = gucd.gucd_modelo_1(1, 0.1, 1, 1, 0.01, 0.01, 2., 3., 1000, 2, 50)
#%%
#Ploteo
plt.figure()
plt.plot(S, A_s, 'o', label='A')
plt.plot(S, B_s, '.', label='B')
plt.grid()
plt.legend()
plt.show()

#%%
#Pruebo la función que mide el área biestable
area = mb.mide_biestabilidad(A_s, S)

#%%
#Pruebo generar parámetros con el lhs
#Me fijo cuánto tarda en generar diez sets de parámetros y hacer todo el análisis
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
            areas.append(area)
    del(A_s, B_s, S, area)
tf = time()
print(f'Tarda {tf -ti} segundos en hacer todo el análisis para 10 sets de parámetros.')

#%%
#Hago lo mismo pero ahora con el dataframe en el que voy a ir guardando los parámetros que den biestabilidad 
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
