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
S_on = df.loc[:, 'S_on'].to_numpy()
S_off = df.loc[:, 'S_off'].to_numpy()

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

#Esta línea ignora los warnings que aparecen, pero el histograma que resulta no esta ok.
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

#Calculo correlación ignorando los casos de memoria 0. Uso solo el coeficiente de Spearman porque tiene más sentido por lo que contó Fede
for par_A, par_B, nombre in zip(ejes_x_A_may, ejes_x_B_may, labels_x):
    r_spearman_A, pv_spearman_A = spearmanr(par_A, mem_A_may[0])
    r_spearman_B, pv_spearman_B = spearmanr(par_B, mem_B_may[0])

    print(f'Para la memoria en A, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_A} con un p-valor de {pv_spearman_A}.')
    print(f'Para la memoria en B, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_B} con un p-valor de {pv_spearman_B}.')

#Para aquellos casos en donde la memoria haya dado menor a 0.1, vuelco a calcularla cambiando ambas cosas que dijo Fede: el valor de S_alto y el valor de la diferencia temporal que la integración toma como constante
#Para todo esto voy a usar que i_mem_A_men = i_mem_B_men
mem_A_salto = np.zeros_like(i_mem_A_men[0])
mem_B_salto = np.zeros_like(i_mem_A_men[0])

mem_A_tes = np.zeros_like(i_mem_A_men[0])
mem_B_tes = np.zeros_like(i_mem_A_men[0])

# for n, i in zip(i_mem_A_men[0], range(len(i_mem_A_men[0]))):
    # params = df.loc[areas[n], :].to_numpy()[:-5]
    # S_on = df.loc[areas[n], :].to_numpy()[-2]
    # S_off = df.loc[areas[n], :].to_numpy()[-1]
    # S_bajo = (S_on + S_off)/2
# 
    # #Aumento el S_alto
    # mem_A_salto[i], mem_B_salto[i] = mm.mide_memoria(*params, S_alto=5, S_bajo=S_bajo, plot_estimulo=False, plot_memoria=False)
# 
# with open(f'mem_A_salto.pkl', 'wb') as f:
            # pickle.dump(mem_A_salto, f)
# with open(f'mem_B_salto.pkl', 'wb') as f:
            # pickle.dump(mem_B_salto, f)

    # #Cambio la diferencia temporal considerada constante
    # mem_A_tes[n], mem_B_tes[n] = mm.mide_memoria(*params, S_alto=5, S_bajo=S_bajo, plot_estimulo=False, plot_memoria=False)
# 
# with open(f'mem_A_tes.pkl', 'wb') as f:
            # pickle.dump(mem_A_tes, f)ç
# with open(f'mem_B_tes.pkl', 'wb') as f:
            # pickle.dump(mem_B_tes, f)
# 
# #Repito los gŕaficos con estos cambios
# with open(f'mem_A_salto.pkl', 'rb') as f:
            # mem_A_salto = pickle.load(f)
# with open(f'mem_B_salto.pkl', 'rb') as f:
            # mem_B_salto = pickle.load(f)

#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_A_may[i], mem_A_may[0], '.c', alpha=0.5)
    ax.plot(ejes_x_A_men[i], mem_A_salto, '.r', alpha=0.5)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en A')
    ax.grid()

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_B_may[i], mem_B_may[0], '.c', alpha=0.5)
    ax.plot(ejes_x_B_men[i], mem_B_salto, '.r', alpha=0.5)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en B')
    ax.grid()

#Veo curva de histéresis de los casos men
#Integro estos sistemas
tiempo_max = 100
S_max = 1
S_min = 0
pasos = 1000

for i in i_mem_A_men[0][:1]:
    params = df.loc[areas[i], :].to_numpy()[:-5]
    A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
    plt.figure()
    plt.plot(S, A_s, 'o', label='A')
    plt.plot(S, B_s, '.', label='B')
    plt.grid()
    plt.legend()
    plt.title(f'Área Biestable: {areas[i]}')
    plt.show()


