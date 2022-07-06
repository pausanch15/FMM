#Analizo los datos obtenidos en 2022_07_04-parametros_biestables-modelo_3.csv

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
fname = '2022_07_04-parametros_biestables-modelo_3.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Memorias medidas en el primer intento, todas juntas
with open(f'2022_07_04-mem_A.pkl', 'rb') as f:
            mem_A = pickle.load(f)                
with open(f'2022_07_04-mem_B.pkl', 'rb') as f:
            mem_B = pickle.load(f)

#De estas ya sé que muchas fallaron. Las saco, estudio cosas en los que sí salieron bien y posteriormente recalculo la memoria de los sistemas que no salieron bien
i_fallados = [0, 7, 8, 13, 16, 17, 28, 29, 33, 35, 40, 41, 46, 52, 55, 56, 57, 60, 62, 63, 64, 65, 68, 73, 80, 89, 91, 92, 93, 96, 104, 107, 109, 115, 125, 134, 135, 141, 145, 146, 149, 156, 159, 160, 168, 169, 176, 179, 184, 185, 186, 198, 203, 205, 208, 217, 218, 221, 223, 224, 225, 230, 231, 232, 233, 236, 237, 239, 243, 245, 247, 250, 254, 260, 262, 263, 264, 266, 280, 283, 284, 285, 286, 289, 291, 295, 296, 298, 299, 304, 307, 311, 314, 317, 324, 325, 327, 328, 329, 333, 334, 344, 351, 354, 357, 363, 366, 377, 380, 382, 383, 387, 389, 391, 392, 400, 402, 403, 409, 410, 411, 414, 415, 424, 426, 428, 429, 434, 437, 438, 440, 441, 442, 445, 446, 449, 452, 454, 455, 457, 459, 463, 464, 468, 470, 476, 477, 478, 481, 491, 495, 496, 497, 498, 501, 502, 504, 506, 508, 510, 512, 513, 520, 525, 526, 530, 531, 532, 533, 537, 541, 542, 543, 544, 545, 551, 552, 556, 557, 558, 559, 563, 565, 566, 570, 580, 584, 585, 589, 590, 591, 596, 597, 600, 605, 616, 621, 624, 625, 629, 630, 639, 344, 348, 352, 356, 660, 661, 663, 664, 667, 671, 675, 677, 679, 682, 684, 685, 698, 699, 702, 708, 710, 721, 730, 731, 733, 737, 738, 742, 743, 745, 751, 756, 757]

mem_A_ok, mem_B_ok = [], []
areas_ok, anchos_ok, altos_on_ok, altos_off_ok = [], [], [], []

mem_A_fallados, mem_B_fallados = [], []
areas_falladas, anchos_fallados, altos_on_fallados, altos_off_fallados = [], [], [], []

for i, area in enumerate(areas):
    if i in i_fallados:
        mem_A_fallados.append(mem_A[i])
        mem_B_fallados.append(mem_B[i])
        anchos_fallados.append(anchos[i])
        altos_on_fallados.append(altos_on[i])
        altos_off_fallados.append(altos_off[i])
        areas_falladas.append(area)
    else:
        mem_A_ok.append(mem_A[i])
        mem_B_ok.append(mem_B[i])
        anchos_ok.append(anchos[i])
        altos_on_ok.append(altos_on[i])
        altos_off_ok.append(altos_off[i])
        areas_ok.append(area)

#%%
#Para los casos no problemáticos:
#Histograma de áreas
plt.figure()
plt.hist(areas_ok, bins='auto', facecolor='c', density=True, stacked=True, edgecolor = "black")
plt.title('Histograma Áreas')
plt.grid(zorder=0)

#Histograma de memoria en A y B
fig, ax = plt.subplots(2, 1)
ax[0].hist(mem_A_ok, bins='auto', facecolor='c', density=True, stacked=True, edgecolor = "black")
ax[0].set_title('Memoria en A')
ax[0].grid()

ax[1].hist(mem_B_ok, bins='auto', facecolor='c', density=True, stacked=True, edgecolor = "black")
ax[1].set_title('Memoria en B')
ax[1].grid()

#Plots
ejes_x = [areas_ok, anchos_ok, altos_on_ok, altos_off_ok]
labels_x = ['Área', 'Ancho', 'Alto On', 'Altos Off']

#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x[i], mem_A_ok, 'co', fillstyle='none')
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en A')
    ax.grid()

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x[i], mem_B_ok, 'co', fillstyle='none')
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
