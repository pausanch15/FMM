#Continúo el análisis que empecé en el v8, pero agrego lo que propuso Fede
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

#Levanto el csv
fname = '2022_02_14-parametros_biestables-modelo_3.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Levanto la memoria de los sistemas, calculada como hasta ahora
with open(f'mem_A.pkl', 'rb') as f:
            mem_A = pickle.load(f)                
with open(f'mem_B.pkl', 'rb') as f:
            mem_B = pickle.load(f)

#Saco los valores que dan memoria infinita
i_mem_A = np.where(mem_A>1e3)
i_mem_B = np.where(mem_B>1e3)

areas = np.delete(areas, i_mem_A[0])
anchos = np.delete(anchos, i_mem_A[0])
altos_on = np.delete(altos_on, i_mem_A[0])
altos_off = np.delete(altos_off, i_mem_A[0])
mem_A = np.delete(mem_A, i_mem_A[0])
mem_B = np.delete(mem_B, i_mem_A[0])

#Histograma de áreas, en log-log
plt.figure()
hist, bins = np.histogram(areas, bins='auto')
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(areas, bins=logbins, facecolor='c', density=True, stacked=True, edgecolor = "black")
plt.xscale('log')
plt.yscale('log')
plt.title('Histograma Áreas')
plt.grid(zorder=0)

#Histograma de memoria en A y B, en log-log

#esta línea ignora los warnings que aparecen, pero el histograma que resulta no esta ok.
# np.seterr(divide='warn', invalid='warn')

# fig, ax = plt.subplots(2, 1)
# 
# hist, bins = np.histogram(mem_A, bins='auto')
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# ax[0].hist(mem_A, bins=logbins, facecolor='c', density=True, stacked=True, edgecolor = "black")
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_title('Memoria en A')
# ax[0].grid()
# 
# hist, bins = np.histogram(mem_B, bins='auto')
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# ax[1].hist(mem_B, bins=logbins, facecolor='c', density=True, stacked=True, edgecolor = "black")
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_title('Memoria en B')
# ax[1].grid()

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
