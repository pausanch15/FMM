#Mido la correlación y hago los histogramas que dijo Ale
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
#Levanto el csv y veo algunos plots
fname = '2022_07_04-parametros_biestables-modelo_3.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Calculo la memoria de los sistemas, si no la calculé antes
if '2022_07_04-mem_A.pkl' in os.listdir() and '2022_07_04-mem_B.pkl' in os.listdir():
    with open(f'2022_07_04-mem_A.pkl', 'rb') as f:
                mem_A = pickle.load(f)                
    with open(f'2022_07_04-mem_B.pkl', 'rb') as f:
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
        plt.savefig(f'resultados/2022_07_04-memoria_{n}.pdf')
        plt.close()

    with open(f'2022_07_04-mem_A.pkl', 'wb') as f:
                pickle.dump(mem_A, f)                
    with open(f'2022_07_04-mem_B.pkl', 'wb') as f:
                pickle.dump(mem_B, f)

    print('Ya guardó los cálculos de memoria')

#%%
#En la memoria hay algunos valores que dan muy altos (ridículamente altos). Reviso esos sistemas
#Encuentro los índices de estos valores. Espero que sean los mismos... lo son
i_mem_A = np.where(mem_A>1e3)
i_mem_B = np.where(mem_B>1e3)

#Integro estos sistemas
# tiempo_max = 100
# S_max = 1
# S_min = 0
# pasos = 1000
# 
# for i in i_mem_A[0]:
    # params = df.loc[areas[i], :].to_numpy()[:-5]
    # A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
    # plt.figure()
    # plt.plot(S, A_s, 'o', label='A')
    # plt.plot(S, B_s, '.', label='B')
    # plt.grid()
    # plt.legend()
    # plt.title(f'Área Biestable: {areas[i]}')
    # plt.show()
# 
    # mm.mide_memoria(*params, S_alto=2, S_bajo=S_bajo, plot_estimulo=True, plot_memoria=True)
    
#Hago los plots que dijo Ale. Saco los casos problemáticos
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

#Histograma de memoria en A y B
fig, ax = plt.subplots(2, 1)
ax[0].hist(mem_A, bins='auto', facecolor='c', density=True, stacked=True, edgecolor = "black")
ax[0].set_title('Memoria en A')
ax[0].grid()

ax[1].hist(mem_B, bins='auto', facecolor='c', density=True, stacked=True, edgecolor = "black")
ax[1].set_title('Memoria en B')
ax[1].grid()

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

#Mido correlación con los coeficientes que propuso Ale
for parametro, nombre in zip(ejes_x, labels_x):
    r_pearson_A, pv_pearson_A = pearsonr(parametro, mem_A)
    r_pearson_B, pv_pearson_B = pearsonr(parametro, mem_B)

    r_spearman_A, pv_spearman_A = spearmanr(parametro, mem_A)
    r_spearman_B, pv_spearman_B = spearmanr(parametro, mem_B)

    print(f'Para la memoria en A, el coeficiente de Pearson entre memoria y {nombre} es {r_pearson_A} con un p-valor de {pv_pearson_A}.')
    print(f'Para la memoria en B, el coeficiente de Pearson entre memoria y {nombre} es {r_pearson_B} con un p-valor de {pv_pearson_B}.')

    print(f'Para la memoria en A, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_A} con un p-valor de {pv_spearman_A}.')
    print(f'Para la memoria en B, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_B} con un p-valor de {pv_spearman_B}.')

    print()
