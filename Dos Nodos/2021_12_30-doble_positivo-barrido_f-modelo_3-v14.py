#Vuelvo a medir la memoriadelos 20000 sistemas que integré (que sé que estánmal,pero para probar) con el código que midela memoriacambiado, a ver qué pasa.

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
import pandas as pd
import mide_memoria_modelo_3 as mm
import os
from glob import glob
import pickle
from scipy.stats import pearsonr, spearmanr
plt.ion()

#%%
#Levanto el csv 
fname = '2022_03_07-parametros_biestables-modelo_3-todos_juntos.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Calculo la memoria de los sistemas, si no la calculé antes
if '2022_05_12-mem_A.pkl' in os.listdir() and '2022_05_12-mem_B.pkl' in os.listdir():
    with open(f'2022_05_12-mem_A.pkl', 'rb') as f:
                mem_A = pickle.load(f)                
    with open(f'2022_05_12-mem_B.pkl', 'rb') as f:
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

        mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=S_bajo, plot_estimulo=False, plot_memoria=False)

    with open(f'2022_05_12-mem_A.pkl', 'wb') as f:
                pickle.dump(mem_A, f)                
    with open(f'2022_05_12-mem_B.pkl', 'wb') as f:
                pickle.dump(mem_B, f)

plt.ioff()
#Veo qué está pasando al calcular memoria
for n, area in enumerate(areas):
    params = df.loc[areas[n], :].to_numpy()[:-5]
    S_on = df.loc[areas[n], :].to_numpy()[-2]
    S_off = df.loc[areas[n], :].to_numpy()[-1]
    S_bajo = (S_on + S_off)/2

    mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=S_bajo, plot_estimulo=True, plot_memoria=True)
    print(f"{n=}")
    plt.show()
plt.ion()

#Saco los valores mayores a 1
i_mem_A = np.where(mem_A>1)
i_mem_B = np.where(mem_B>1)

areas = np.delete(areas, i_mem_B[0])
anchos = np.delete(anchos, i_mem_B[0])
altos_on = np.delete(altos_on, i_mem_B[0])
altos_off = np.delete(altos_off, i_mem_B[0])
mem_A = np.delete(mem_A, i_mem_B[0])
mem_B = np.delete(mem_B, i_mem_B[0])

#Histograma de áreas
plt.figure()
plt.hist(areas, bins='auto', facecolor='c', density=True, stacked=True, edgecolor = "black")
plt.title('Histograma Áreas')
plt.grid(zorder=0)

#Plots
ejes_x = [areas, anchos, altos_on, altos_off]
labels_x = ['Área', 'Ancho', 'Alto On', 'Altos Off']

#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x[i], mem_A, 'co', fillstyle='none')
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en A')
    ax.grid()

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x[i], mem_B, 'co', fillstyle='none')
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en B')
    ax.grid()

plt.show()
