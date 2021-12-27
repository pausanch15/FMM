#Acá es donde voy a hacer el barrido eligiendo los parámetros con el lhs para la nueva forma de filtrar
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v2 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_1 as gucd
import latin_hypercube_sampling as lhs
import pandas as pd
from tqdm import tqdm
plt.ion()

#%%
#El análisis
df = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb'])

n_parametros = 8
n_barrido = 10000
parametros = lhs.lhs(n_parametros, n_barrido)

areas = []

for params, i in zip(parametros[2808:], tqdm(range(2808, n_barrido))):
    try:
        A_s, B_s, S = gucd.gucd_modelo_1(*params, 1000, 2, 50)
        area = mb.mide_biestabilidad(A_s, S)
        if area != None:
            if area > 0:
                df1 = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb'], index=[str(area)])
                df1.loc[str(area), :] = params
                df = df.append(df1)
                del(df1)
        del(A_s, B_s, S, area)
    except:
        pass
    
df.to_csv('2021_12_24-parametros_biestables.csv')

#%%
#Levanto el csv y veo algunos plots
fname = '2021_12_24-parametros_biestables.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()

#Elijo algún conjunto de parámetros e integro el modelo para ellos
n = 50
params = df.loc[areas[n], :].to_numpy()
A_s, B_s, S = gucd.gucd_modelo_1(*params, 1000, 2, 50)

#Ploteo
plt.figure()
plt.plot(S, A_s, 'o', label='A')
plt.plot(S, B_s, '.', label='B')
plt.grid()
plt.legend()
plt.show()
