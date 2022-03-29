#Acá trato de hacer la distribución de memoria para los distintos parámetros de las ecuaciones del modelo.

#Librerías 
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
import latin_hypercube_sampling as lhs
import pandas as pd
import mide_memoria_modelo_3 as mm
import os
import pickle
from multiprocessing import Pool
from itertools import combinations
import random
plt.ion()

#Levanto el csv y veo algunos plots
fname = '2022_03_07-parametros_biestables-modelo_3-todos_juntos.csv'
df = pd.read_csv(fname, index_col=0)

#Armo los arrays con las cantidades importantes
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()
S_on = df.loc[:, 'S_on'].to_numpy()
S_off = df.loc[:, 'S_off'].to_numpy()

#Armo los arrays de memoria juntando los que ya calculé
with open(f'2022_02_14-mem_A.pkl', 'rb') as f:
            mem_A_1 = pickle.load(f)                
with open(f'2022_02_14-mem_B.pkl', 'rb') as f:
            mem_B_1 = pickle.load(f)

with open(f'2022_03_07-mem_A.pkl', 'rb') as f:
            mem_A_2 = pickle.load(f)                
with open(f'2022_03_07-mem_B.pkl', 'rb') as f:
            mem_B_2 = pickle.load(f)

mem_A = np.concatenate((mem_A_1, mem_A_2))
mem_B = np.concatenate((mem_B_1, mem_B_2))
    
#Saco los valores que dan memoria infinita 
i_mem_A = np.where(mem_A>1e3)
i_mem_B = np.where(mem_B>1e3)

areas = np.delete(areas, i_mem_B[0])
anchos = np.delete(anchos, i_mem_B[0])
altos_on = np.delete(altos_on, i_mem_B[0])
altos_off = np.delete(altos_off, i_mem_B[0])
mem_A = np.delete(mem_A, i_mem_B[0])
mem_B = np.delete(mem_B, i_mem_B[0])

#Empiezo a hacer los pcolormesh. Me copio las ecuaciones para saber qué traer
# dA = S*k_sa*(1-A)/(K_sa+1-A) + B*k_ba*(1-A)/(K_ba+1-A) - k_ba*A/(K_ba+A)
# dB = S*k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
#En total son 28 combinaciones posibles
parametros = df.columns.to_numpy()[:-5]
pares = list(combinations(parametros, 2))

K_sa_s = np.delete(df.loc[:, parametros[0]].to_numpy(), i_mem_B[0])
K_sb_s = np.delete(df.loc[:, parametros[1]].to_numpy(), i_mem_B[0])
k_ba_s = np.delete(df.loc[:, parametros[2]].to_numpy(), i_mem_B[0])
k_ab_s = np.delete(df.loc[:, parametros[3]].to_numpy(), i_mem_B[0])
K_ba_s = np.delete(df.loc[:, parametros[4]].to_numpy(), i_mem_B[0])
K_ab_s = np.delete(df.loc[:, parametros[5]].to_numpy(), i_mem_B[0])
k_sa_s = np.delete(df.loc[:, parametros[6]].to_numpy(), i_mem_B[0])
k_sb_s = np.delete(df.loc[:, parametros[7]].to_numpy(), i_mem_B[0])

parametros_num = [K_sa_s, K_sb_s, k_ba_s, k_ab_s, K_ba_s, K_ab_s, k_sa_s, k_sb_s]
pares_num = list(combinations(parametros_num, 2))

#Ejemplo particular
x_max, x_min = np.max(k_sa_s), np.min(k_sa_s)
y_max, y_min = np.max(k_sb_s), np.min(k_sb_s)
x, y = np.meshgrid(np.linspace(x_min, x_max, 59), np.linspace(y_min, y_max, 10))
z = np.reshape(mem_A, np.shape(x))
plt.figure()
plt.pcolor(x, y, z, cmap='RdBu')
plt.colorbar()
plt.show()
