#Ya estudié los dos conjuntos de 10000 corridas y ambos se comportan estadísticamente igual, por lo que junto todos los datos que tengo, obtengo gráficos y recalculo la memoria en aquellos casos en los que me dio 0 porque el S_bajo estaba mal elegido. Los S_bajo nuevos los elegí viendo las curvas de histéresis en cada uno de los casos. 

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
plt.ion()

#Junto ambos dataframes en uno solo, o lo traigo si ya está hecho
fname = '2022_03_07-parametros_biestables-modelo_3-todos_juntos.csv'
if fname in os.listdir():
    #Levanto el csv y veo algunos plots
    df = pd.read_csv(fname, index_col=0)
    
else:
    df1 = pd.read_csv('2022_02_14-parametros_biestables-modelo_3.csv', index_col=0)
    df2 = pd.read_csv('2022_03_07-parametros_biestables-modelo_3.csv', index_col=0)
    df = pd.concat([df1,df2])
    df.to_csv(fname)
    del(df1, df2)
    
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

#Histograma de áreas, en log-log
plt.figure()
hist, bins = np.histogram(areas, bins='auto')
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(areas, bins=logbins, facecolor='c', density=True, stacked=True, edgecolor = "black")
plt.xscale('log')
plt.yscale('log')
plt.title('Histograma Áreas')
plt.grid(zorder=0)

#Rehago los gráficos que ven correlación pintando los valores de memoria menores a 0.1 de otro color, y les pongo un alpha para ver la densidad de puntos en el gráfico
ejes_x = [areas, anchos, altos_on, altos_off]
labels_x = ['Área', 'Ancho', 'Alto On', 'Altos Off']

i_mem_A_may = np.where(mem_A>=0.1)
i_mem_A_men = np.where(mem_A<0.1)

i_mem_B_may = np.where(mem_A>=0.1)
i_mem_B_men = np.where(mem_A<0.1)

mem_A_may = [mem_A[i] for i in i_mem_A_may]
mem_A_men = [mem_A[i] for i in i_mem_A_men]

mem_B_may = [mem_B[i] for i in i_mem_B_may]
mem_B_men = [mem_B[i] for i in i_mem_B_men]

ejes_x_A_may = [arr[i] for arr in ejes_x for i in i_mem_A_may]
ejes_x_A_men = [arr[i] for arr in ejes_x for i in i_mem_A_men]

ejes_x_B_may = [arr[i] for arr in ejes_x for i in i_mem_B_may]
ejes_x_B_men = [arr[i] for arr in ejes_x for i in i_mem_B_men]

#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_A_may[i], mem_A_may[0], '.c', alpha=0.5)
    ax.plot(ejes_x_A_men[i], mem_A_men[0], '.r', alpha=0.5)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en A')
    ax.grid()

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_B_may[i], mem_B_may[0], '.c', alpha=0.5)
    ax.plot(ejes_x_B_men[i], mem_B_men[0], '.r', alpha=0.5)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en B')
    ax.grid()

plt.show()

#Recalculo la memoria en aquellos casos donde me haya dado menor a 0.1 con un S_bajo cambiado y un S_alto mayor
# S_bajos_nuevos_1 = [0., 0.17, 0.29, 0.19, 0.58, 0.207, 0.46, 0.43, 0.45, 0.13, 0.09, float("nan"), 0.15, 0.2, 0.42, 0.04, 0.07, 0.04, 0.8, 0.4, float("nan"), 0.2, 0.4, 0.03, 0.7, 0.35, 0.19]
# 
# S_bajos_nuevos_2 = [float("nan"), 0.01, 0.62, 0.25, 0.8, 0.2, 0.7, 0.4, 0.2, 0.3, 0.45, 0.25, 0.27, 0.25, float("nan"), float("nan"), 0.2, 0.22, 0.27, float("nan"), 0.8, 0.4, float("nan"), 0.22, 0.6, float("nan"), 0.25, 0.195]

S_bajos_nuevos = [0., 0.17, 0.29, 0.19, 0.58, 0.207, 0.46, 0.43, 0.45, 0.13, 0.09, float("nan"), 0.15, 0.2, 0.42, 0.04, 0.07, 0.04, 0.8, 0.4, float("nan"), 0.2, 0.4, 0.03, 0.7, 0.35, 0.19, float("nan"), 0.01, 0.62, 0.25, 0.8, 0.2, 0.7, 0.4, 0.2, 0.3, 0.45, 0.25, 0.27, 0.25, float("nan"), float("nan"), 0.2, 0.22, 0.27, float("nan"), 0.8, 0.4, float("nan"), 0.22, 0.6, float("nan"), 0.25, 0.195]

if '2022_03_08-mem_A_men_nuevas.pkl' in os.listdir() and '2022_03_08-mem_B_men_nuevas.pkl' in os.listdir():
    with open(f'2022_03_08-mem_A_men_nuevas.pkl', 'rb') as f:
                mem_A_men_nuevas = pickle.load(f)                
    with open(f'2022_03_08-mem_B_men_nuevas.pkl', 'rb') as f:
                mem_B_men_nuevas = pickle.load(f)

else:
    mem_A_men_nuevas = np.zeros_like(S_bajos_nuevos)
    mem_B_men_nuevas = np.zeros_like(S_bajos_nuevos)

    for n, S_bajo, i in zip(i_mem_A_men[0], S_bajos_nuevos, range(len(S_bajos_nuevos))):
        if not np.isnan(S_bajo):
            print(f'Va por S_bajo={S_bajo}')
            params = df.loc[areas[n], :].to_numpy()[:-5]
            mem_A_men_nuevas[i], mem_B_men_nuevas[i] = mm.mide_memoria(*params, S_alto=10, S_bajo=S_bajo, plot_estimulo=False, plot_memoria=False)
        else:
            mem_A_men_nuevas[i], mem_B_men_nuevas[i] = float("nan"), float("nan")

    with open(f'2022_03_08-mem_A_men_nuevas.pkl', 'wb') as f:
                pickle.dump(mem_A_men_nuevas, f)                
    with open(f'2022_03_08-mem_B_men_nuevas.pkl', 'wb') as f:
                pickle.dump(mem_B_men_nuevas, f)

#Grafico nuevamente para ver cuánto dieron estos valores ahora
#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_A_may[i], mem_A_may[0], '.c', alpha=0.5)
    ax.plot(ejes_x_A_men[i], mem_A_men_nuevas, '.r', alpha=0.5)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en A')
    ax.set_ylim(-0.05, 1.05) #Hago esto porque algunos valores dan infinito
    ax.grid()

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_B_may[i], mem_B_may[0], '.c', alpha=0.5)
    ax.plot(ejes_x_B_men[i], mem_B_men_nuevas, '.r', alpha=0.5)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en B')
    ax.set_ylim(-0.05, 1.05) #Hago esto porque algunos valores dan infinito
    ax.grid()

plt.show()
