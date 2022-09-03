#Calculo la memoria de los sistemas que encontre en v2 y estan en el csv que crea el v3

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_negativo_v1 as gucd
import pandas as pd
import mide_memoria_doble_negativo as mm
import os
from glob import glob
import pickle
from scipy.stats import pearsonr, spearmanr
plt.ion()

#%%
#Levanto el csv y veo algunos plots
fname = '2022_08_31-parametros_biestables-doble_negativo.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Calculo la memoria de los sistemas, si no la calculé antes
if '2022_08_31-mem_A.pkl' in os.listdir() and '2022_08_31-mem_B.pkl' in os.listdir():
    with open(f'2022_08_31-mem_A.pkl', 'rb') as f:
                mem_A = pickle.load(f)                
    with open(f'2022_08_31-mem_B.pkl', 'rb') as f:
                mem_B = pickle.load(f)

else:
    mem_A = np.zeros_like(areas)
    mem_B = np.zeros_like(areas)

    for n, area in enumerate(areas):
        if n%10 == 0: print(n)
        params = df.loc[areas[n], :].to_numpy()[:-5]
        S_on = df.loc[areas[n], :].to_numpy()[-2]
        S_off = df.loc[areas[n], :].to_numpy()[-1]
        S_bajo = (S_on + S_off)/2

        mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=S_bajo, plot_mem=True, plot_est=False)

        #Guardo la figura de la memoria
        plt.savefig(f'resultados/2022_08_31-memoria_{n}.pdf')
        plt.close()

    with open(f'2022_08_31-mem_A.pkl', 'wb') as f:
                pickle.dump(mem_A, f)                
    with open(f'2022_08_31-mem_B.pkl', 'wb') as f:
                pickle.dump(mem_B, f)

    print('Ya guardó los cálculos de memoria')
