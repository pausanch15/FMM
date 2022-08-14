#Todo el anlálisis para los sistemas cuya memoria se midió bien. En el 2022_07_04-parametros_biestables-modelo_3.csv hay 477 casos para analizar. En el 2022_07_25-parametros_biestables-modelo_3.csv hay 
#Acá hago el análisis para las corridas que tengo juntas. En v18 están para una a la vez

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import runge_kuta_estimulo as rks
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
import pandas as pd
import mide_memoria_modelo_3 as mm
import os
from glob import glob
import pickle
from scipy.stats import pearsonr, spearmanr
from scipy import interpolate
import random
from itertools import combinations
import latin_hypercube_sampling as lhs
plt.style.use('seaborn-pastel')
plt.ion()

#%%
#Levanto el csv 
fname = '2022_07_25-parametros_biestables-modelo_3-todos_juntos.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Memorias medidas en el primer intento, todas juntas
with open(f'2022_07_04-mem_A.pkl', 'rb') as f:
            mem_A_04 = pickle.load(f)                
with open(f'2022_07_04-mem_B.pkl', 'rb') as f:
            mem_B_04 = pickle.load(f)

with open(f'2022_07_25-mem_A.pkl', 'rb') as f:
            mem_A_25 = pickle.load(f)                
with open(f'2022_07_25-mem_B.pkl', 'rb') as f:
            mem_B_25 = pickle.load(f)

mem_A = np.concatenate((mem_A_04, mem_A_25))
mem_B = np.concatenate((mem_B_04, mem_B_25))

#De estas ya sé que muchas fallaron. Las saco, estudio cosas en los que sí salieron bien y posteriormente recalculo la memoria de los sistemas que no salieron bien

#Corrida del 2022_07_04
# i_fallados = [0, 7, 8, 13, 16, 17, 28, 29, 33, 35, 40, 41, 46, 52, 55, 56, 57, 60, 62, 63, 64, 65, 68, 73, 80, 89, 91, 92, 93, 96, 104, 107, 109, 115, 125, 134, 135, 141, 145, 146, 149, 156, 159, 160, 168, 169, 176, 179, 184, 185, 186, 198, 203, 205, 208, 217, 218, 221, 223, 224, 225, 230, 231, 232, 233, 236, 237, 239, 243, 245, 247, 250, 254, 260, 262, 263, 264, 266, 280, 283, 284, 285, 286, 289, 291, 295, 296, 298, 299, 304, 307, 311, 314, 317, 324, 325, 327, 328, 329, 333, 334, 344, 351, 354, 357, 363, 366, 377, 380, 382, 383, 387, 389, 391, 392, 400, 402, 403, 409, 410, 411, 414, 415, 424, 426, 428, 429, 434, 437, 438, 440, 441, 442, 445, 446, 449, 452, 454, 455, 457, 459, 463, 464, 468, 470, 476, 477, 478, 481, 491, 495, 496, 497, 498, 501, 502, 504, 506, 508, 510, 512, 513, 520, 525, 526, 530, 531, 532, 533, 537, 541, 542, 543, 544, 545, 551, 552, 556, 557, 558, 559, 563, 565, 566, 570, 580, 584, 585, 589, 590, 591, 596, 597, 600, 605, 616, 621, 624, 625, 629, 630, 639, 344, 348, 352, 356, 660, 661, 663, 664, 667, 671, 675, 677, 679, 682, 684, 685, 698, 699, 702, 708, 710, 721, 730, 731, 733, 737, 738, 742, 743, 745, 751, 756, 757]

#Corrida 2022_07_25
# i_fallados = [0, 4, 5, 6, 7, 15, 17, 18, 24, 25, 26, 29, 30, 31, 34, 37, 39, 45, 49, 54, 55, 56, 60, 62, 64, 67, 68, 69, 71, 73, 75, 77, 79, 80, 81, 86, 89, 90, 96, 100, 101, 103, 105, 108, 116, 120, 121, 122, 125, 127, 131, 133, 138, 140, 146, 147, 148, 150, 151, 153, 154, 158, 167, 168, 171, 173, 174, 182, 183, 187, 189, 192, 200, 204, 205, 208, 217, 218, 220, 224, 226, 227, 228, 229, 231, 235, 236, 239, 240, 241, 247, 249, 251, 252, 255, 256, 257, 258, 261, 262, 263, 266, 267, 271, 273, 276, 277, 283, 285, 286, 288, 293, 294, 295, 297, 301, 306, 307, 308, 311, 314, 316, 318, 319, 322, 324, 325, 327, 329, 330, 331, 332, 333, 335, 339, 348, 351, 357, 359, 363, 364, 365, 373, 376, 381, 382, 384, 385, 389, 394, 397, 399, 400, 408, 410, 411, 413, 417, 421, 430, 432, 438, 439, 441, 447, 450, 452, 456, 457, 461, 464, 468, 469, 473, 476, 477, 479, 480, 486, 488, 491, 495, 497, 499, 505, 511, 512, 516, 517, 518, 519, 521, 522, 523, 525, 526, 533, 534, 537, 543, 545, 548, 554, 558, 559, 561, 563, 564, 567, 569, 572, 573, 576, 578, 579, 587, 589, 590, 597, 598, 600, 601, 602, 603, 605, 606, 617, 619, 620, 622, 624, 626, 627, 629, 630, 631, 634, 637, 643, 648, 649, 651, 652, 656, 659, 662, 663, 664, 667, 668, 674, 676, 677, 682, 683, 694, 695, 696, 699, 700, 701, 702, 704, 705, 708, 711, 713, 715, 717, 719, 722, 724, 726, 727, 728, 729, 733, 734, 738, 745, 746, 747, 748, 752, 758, 759, 762, 764, 766, 767, 769, 771, 773, 777, 780, 782, 784, 787, 791, 793]

#Corridas de 2022_07_04 y 2022_07_25 juntas
i_fallados_04 = np.array([0, 7, 8, 13, 16, 17, 28, 29, 33, 35, 40, 41, 46, 52, 55, 56, 57, 60, 62, 63, 64, 65, 68, 73, 80, 89, 91, 92, 93, 96, 104, 107, 109, 115, 125, 134, 135, 141, 145, 146, 149, 156, 159, 160, 168, 169, 176, 179, 184, 185, 186, 198, 203, 205, 208, 217, 218, 221, 223, 224, 225, 230, 231, 232, 233, 236, 237, 239, 243, 245, 247, 250, 254, 260, 262, 263, 264, 266, 280, 283, 284, 285, 286, 289, 291, 295, 296, 298, 299, 304, 307, 311, 314, 317, 324, 325, 327, 328, 329, 333, 334, 344, 351, 354, 357, 363, 366, 377, 380, 382, 383, 387, 389, 391, 392, 400, 402, 403, 409, 410, 411, 414, 415, 424, 426, 428, 429, 434, 437, 438, 440, 441, 442, 445, 446, 449, 452, 454, 455, 457, 459, 463, 464, 468, 470, 476, 477, 478, 481, 491, 495, 496, 497, 498, 501, 502, 504, 506, 508, 510, 512, 513, 520, 525, 526, 530, 531, 532, 533, 537, 541, 542, 543, 544, 545, 551, 552, 556, 557, 558, 559, 563, 565, 566, 570, 580, 584, 585, 589, 590, 591, 596, 597, 600, 605, 616, 621, 624, 625, 629, 630, 639, 344, 348, 352, 356, 660, 661, 663, 664, 667, 671, 675, 677, 679, 682, 684, 685, 698, 699, 702, 708, 710, 721, 730, 731, 733, 737, 738, 742, 743, 745, 751, 756, 757])

i_fallados_25 = np.array([0, 4, 5, 6, 7, 15, 17, 18, 24, 25, 26, 29, 30, 31, 34, 37, 39, 45, 49, 54, 55, 56, 60, 62, 64, 67, 68, 69, 71, 73, 75, 77, 79, 80, 81, 86, 89, 90, 96, 100, 101, 103, 105, 108, 116, 120, 121, 122, 125, 127, 131, 133, 138, 140, 146, 147, 148, 150, 151, 153, 154, 158, 167, 168, 171, 173, 174, 182, 183, 187, 189, 192, 200, 204, 205, 208, 217, 218, 220, 224, 226, 227, 228, 229, 231, 235, 236, 239, 240, 241, 247, 249, 251, 252, 255, 256, 257, 258, 261, 262, 263, 266, 267, 271, 273, 276, 277, 283, 285, 286, 288, 293, 294, 295, 297, 301, 306, 307, 308, 311, 314, 316, 318, 319, 322, 324, 325, 327, 329, 330, 331, 332, 333, 335, 339, 348, 351, 357, 359, 363, 364, 365, 373, 376, 381, 382, 384, 385, 389, 394, 397, 399, 400, 408, 410, 411, 413, 417, 421, 430, 432, 438, 439, 441, 447, 450, 452, 456, 457, 461, 464, 468, 469, 473, 476, 477, 479, 480, 486, 488, 491, 495, 497, 499, 505, 511, 512, 516, 517, 518, 519, 521, 522, 523, 525, 526, 533, 534, 537, 543, 545, 548, 554, 558, 559, 561, 563, 564, 567, 569, 572, 573, 576, 578, 579, 587, 589, 590, 597, 598, 600, 601, 602, 603, 605, 606, 617, 619, 620, 622, 624, 626, 627, 629, 630, 631, 634, 637, 643, 648, 649, 651, 652, 656, 659, 662, 663, 664, 667, 668, 674, 676, 677, 682, 683, 694, 695, 696, 699, 700, 701, 702, 704, 705, 708, 711, 713, 715, 717, 719, 722, 724, 726, 727, 728, 729, 733, 734, 738, 745, 746, 747, 748, 752, 758, 759, 762, 764, 766, 767, 769, 771, 773, 777, 780, 782, 784, 787, 791, 793])+len(mem_A_04)

i_fallados = np.concatenate((i_fallados_04, i_fallados_25))

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
#Histograma de áreas, en log-log
plt.figure()
hist, bins = np.histogram(areas_ok, bins='auto')
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(areas_ok, bins=logbins, facecolor='c', density=True, stacked=True, edgecolor = "black")
plt.xscale('log')
plt.yscale('log')
plt.title('Histograma Áreas')
plt.grid(zorder=0)

#%%
#Gráficos que ven correlación pintando los valores de memoria menores a 0.1 de otro color, y les pongo un alpha para ver la densidad de puntos en el gráfico
mem_A_ok = np.array(mem_A_ok)
mem_B_ok = np.array(mem_B_ok)

ejes_x = [areas_ok, anchos_ok, altos_on_ok, altos_off_ok]
labels_x = ['Área', 'Ancho', 'Alto On', 'Altos Off']

i_mem_A_may = np.where(mem_A_ok>=0.1)
i_mem_A_men = np.where(mem_A_ok<0.1)

i_mem_B_may = np.where(mem_A_ok>=0.1)
i_mem_B_men = np.where(mem_A_ok<0.1)

mem_A_may = [mem_A_ok[i] for i in i_mem_A_may[0]]
mem_A_men = [mem_A_ok[i] for i in i_mem_A_men[0]]

mem_B_may = [mem_B_ok[i] for i in i_mem_B_may[0]]
mem_B_men = [mem_B_ok[i] for i in i_mem_B_men[0]]

ejes_x_A_may, ejes_x_A_men, ejes_x_B_may, ejes_x_B_men = [], [], [], []

ejes = [ejes_x_A_may, ejes_x_A_men, ejes_x_B_may, ejes_x_B_men]
i_mems = [i_mem_A_may, i_mem_A_men, i_mem_B_may, i_mem_B_men]

for e, ej in enumerate(ejes):
    for arr in ejes_x:
        eje = []
        for i in i_mems[e][0]:
            eje.append(arr[i])
        ej.append(eje)
        del(eje)

#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_A_may[i], mem_A_may, '.c', alpha=0.3)
    ax.plot(ejes_x_A_men[i], mem_A_men, '.r', alpha=0.3)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en A')
    ax.grid()

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_B_may[i], mem_B_may, '.c', alpha=0.3)
    ax.plot(ejes_x_B_men[i], mem_B_men, '.r', alpha=0.3)
    ax.set_xlabel(labels_x[i])
    ax.set_ylabel('Memoria en B')
    ax.grid()

plt.show()

#Figura de correlación entre las memorias mayores de ambas variables y los valores de altos on y off
#Altos On
plt.figure()
plt.plot(ejes_x_A_may[-2], mem_A_may, '.', label='A: rSpearman=0.73')
plt.plot(ejes_x_B_may[-2], mem_B_may, '.', label='B: rSpearman=0.41')
plt.xlabel(labels_x[-2])
plt.ylabel('Memoria')
plt.legend()
plt.grid()

#Altos Off
plt.figure()
plt.plot(ejes_x_A_may[-1], mem_A_may, '.', label='A: rSpearman=0.86')
plt.plot(ejes_x_B_may[-1], mem_B_may, '.', label='B: rSpearman=0.67')
plt.xlabel(labels_x[-1])
plt.ylabel('Memoria')
plt.legend()
plt.grid()

#%%
#Calculo correlación ignorando los casos de memoria 0. Uso solo el coeficiente de Spearman porque tiene más sentido por lo que contó Fede
for par_A, par_B, nombre in zip(ejes_x_A_may, ejes_x_B_may, labels_x):
    r_spearman_A, pv_spearman_A = spearmanr(par_A, mem_A_may)
    r_spearman_B, pv_spearman_B = spearmanr(par_B, mem_B_may)

    print(f'Para la memoria en A, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_A} con un p-valor de {pv_spearman_A}.')
    print(f'Para la memoria en B, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_B} con un p-valor de {pv_spearman_B}.')

#%%
#Distribución de memoria para los distintos parámetros de las ecuaciones del modelo.
parametros = df.columns.to_numpy()[:-5]
pares = list(combinations(parametros, 2))

K_sa_s = np.delete(df.loc[:, parametros[0]].to_numpy(), i_fallados)
K_sb_s = np.delete(df.loc[:, parametros[1]].to_numpy(), i_fallados)
k_ba_s = np.delete(df.loc[:, parametros[2]].to_numpy(), i_fallados)
k_ab_s = np.delete(df.loc[:, parametros[3]].to_numpy(), i_fallados)
K_ba_s = np.delete(df.loc[:, parametros[4]].to_numpy(), i_fallados)
K_ab_s = np.delete(df.loc[:, parametros[5]].to_numpy(), i_fallados)
k_sa_s = np.delete(df.loc[:, parametros[6]].to_numpy(), i_fallados)
k_sb_s = np.delete(df.loc[:, parametros[7]].to_numpy(), i_fallados)

parametros_num = [K_sa_s, K_sb_s, k_ba_s, k_ab_s, K_ba_s, K_ab_s, k_sa_s, k_sb_s]
pares_num = list(combinations(parametros_num, 2))

#Hago todas las combinaciones posibles
for par, par_num in zip(pares, pares_num):
    plt.figure()
    #Scatter para hacer el plot XYZ de a pares de parametros con la memoria respectiva
    plt.scatter(*par_num, c=mem_A_ok, marker=".",cmap="cividis")    
    plt.colorbar()
    plt.xlabel(par[0])
    plt.ylabel(par[1])
    # plt.savefig(f'resultados/2022_03_30-{par[0]}_{par[1]}.pdf')
    plt.show()
    # plt.close()

#%%
#Histogramas de los parametros
n_parametros = 8
n_barrido = 10000
parame = np.concatenate((lhs.lhs(n_parametros, n_barrido), lhs.lhs(n_parametros, n_barrido, random_seed=1)))

#Elegimos entre que valores queremos la distribucion
parametros = np.copy(parame)

#Permito solo que los k chiquitos se muevan entre 10**(-2) y 10**2
parametros[:, 2] = 10**((parame[:,2]-0.5)*4)
parametros[:, 3] = 10**((parame[:,3]-0.5)*4)
parametros[:, 6] = 10**((parame[:,6]-0.5)*4)
parametros[:, 7] = 10**((parame[:,7]-0.5)*4)

#Y que que los K grandes se muevan entre entre 0.01 y 1
parametros[:, 0] = 10**((parame[:,0]-1)*2)
parametros[:, 1] = 10**((parame[:,1]-1)*2)
parametros[:, 4] = 10**((parame[:,4]-1)*2)
parametros[:, 5] = 10**((parame[:,5]-1)*2)

#Reacomodo los parámetros para poder recorrerlos por tipo de parámetro y no por sistema
parametros_todos = parametros.T

#Dejo anotado el orden que tiene que tener la lista siempre
# ['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On', S_on. S_off]

#Ahora sí, los gráficos. En escala log-log
for col, param in zip(df, parametros_todos):
    if col in df.columns.to_numpy()[8:]: continue
    plt.figure()
    plt.xscale('log')
    parametro = df.loc[:, col].to_numpy()

    hist, bins = np.histogram(param, bins=50)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(param, bins=logbins, facecolor='r', edgecolor="black", alpha=0.4, label='Todos')

    hist, bins = np.histogram(parametro, bins=50)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(parametro, bins=logbins, facecolor='c', edgecolor="black", alpha=0.4, label='Sistemas Biestables')
    
    plt.xlabel(f"{col}")
    plt.ylabel("# Sistemas")
    plt.legend()
    # plt.savefig(f"resultados/2022_04_11-histogramas_log_{col}.pdf")
    # plt.close()

#Separando por k chiquitos y K grandes
k_chicos = ['k_ba', 'k_ab', 'k_sa', 'k_sb']
K_grandes =['K_sa', 'K_sb', 'K_ba', 'K_ab']

#k chicos
plt.figure()
plt.xscale('log')
plt.title('k Chicos')
for col in df:
    if col in df.columns.to_numpy()[8:]: continue
    if str(col) in k_chicos:
        parametro = df.loc[:, col].to_numpy()

        hist, bins = np.histogram(parametro, bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(parametro, bins=logbins, edgecolor="black", alpha=0.4, label=f"{col}")

plt.ylabel("# Sistemas")
plt.legend()

#K grandes
plt.figure()
plt.xscale('log')
plt.title('K Grandes')
for col in df:
    if col in df.columns.to_numpy()[8:]: continue
    if str(col) in K_grandes:
        parametro = df.loc[:, col].to_numpy()

        hist, bins = np.histogram(parametro, bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(parametro, bins=logbins, edgecolor="black", alpha=0.4, label=f"{col}")

plt.ylabel("# Sistemas")
plt.legend()
