#Hago el análisis para los sistemas biestables que encontré
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

#Cosas de matplotlib para hacer los gráficos
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ggplot')
plt.rc("text", usetex=True)
plt.rc('font', family='serif')
plt.ion()

#%%
#Levanto el csv 
fname = '2022_08_31-parametros_biestables-doble_negativo.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()

#Memorias medidas en el primer intento, todas juntas
with open(f'2022_08_31-mem_A.pkl', 'rb') as f:
            mem_A = pickle.load(f)                
with open(f'2022_08_31-mem_B.pkl', 'rb') as f:
            mem_B = pickle.load(f)

#%%
#De estas ya sé que muchas fallaron. Las saco, estudio cosas en los que sí salieron bien y posteriormente recalculo la memoria de los sistemas que no salieron bien
i_fallados = [0, 4, 6, 7, 11, 13, 15, 22, 23, 26, 29, 36, 41, 42, 44, 45, 48, 50, 54, 57, 59, 68, 71, 72, 73, 74, 75, 76, 80, 86, 88, 90, 93, 95, 99, 103, 105, 108, 109, 110, 111, 115, 117, 118, 125, 127, 130, 134, 136, 137, 139, 140, 145, 146, 151, 152, 154, 158, 160, 161, 163, 166, 167, 175, 180, 182, 183, 185, 186, 187, 189, 192, 193, 195, 198, 202, 203, 204, 205, 206, 209, 210, 216, 217, 220, 225, 226, 228, 229, 230, 232, 235, 238, 241, 255, 257, 258, 259, 260, 261, 267, 271, 273, 276, 278, 281, 282, 287, 288, 294, 298, 305, 320, 322, 323, 325, 327, 328, 329, 330, 331, 332, 336, 339, 340, 341, 343, 344, 345, 346, 347, 349, 350, 351, 356, 357, 358, 361, 367, 372, 375, 379, 380, 383, 384, 385, 389, 391, 393, 394, 398, 403, 404, 407, 409, 412, 413, 414, 415, 416, 420, 424, 425, 427, 432, 433, 434, 436, 440, 443, 444, 453, 455, 457, 459, 461, 463, 466, 467, 468, 471, 473, 474, 475, 476, 477, 479, 480, 485, 486, 490, 493, 494, 507, 508, 511, 512, 516, 517, 521, 522, 523, 524, 526, 527, 528, 536, 537, 540, 541, 542, 547, 549, 550, 551, 557, 560, 565, 570, 572, 574, 577, 578, 581, 591, 592, 594, 596, 600, 601, 605, 606, 607, 608, 609, 610, 614, 616, 617, 619, 623, 624, 628, 633, 634, 635, 636, 641, 644, 645, 653, 654, 655, 656, 659, 661, 664, 667, 668, 670, 677, 678, 682, 684, 686, 688, 691, 693, 694, 695, 699, 700, 702, 703, 706, 707, 712, 714, 719, 720, 722, 723, 725, 726, 728, 733, 737, 738, 739, 740, 745, 747, 748, 749, 752, 753, 759, 765, 771, 773, 774, 775, 776, 777, 781, 782, 783, 792, 793, 796, 798, 799, 801, 803, 806, 808, 811, 813, 815, 818, 820, 822, 825, 826, 828, 830, 832, 833, 837, 850, 852, 853, 854, 855, 859, 864, 865, 868, 871, 872, 873, 875, 881, 882, 886, 890, 900, 905, 912, 913, 917, 920, 926, 927, 929, 933, 935, 936, 941, 942, 950, 951, 953, 955, 956, 957]

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
plt.hist(areas_ok, bins=logbins, density=True, stacked=True, alpha=0.3, color='#E24A33')
hist, bins = np.histogram(areas_ok, bins=logbins, density=True)
plt.step(bins[:-1], hist, where='post', color='#E24A33')
plt.xscale('log')
plt.yscale('log')
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('Áreas', fontsize=15, color='black')
plt.ylabel('Frecuencias', fontsize=15, color='black')
plt.grid(True)
plt.tight_layout()
# plt.savefig('Figuras/dobleneghistareasposloglog.pdf', dpi=300, transparent=True)

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

#%%
#Para la memoria en A
fig, axs = plt.subplots(2, 2, sharey=True, figsize=(7, 7))
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_A_may[i], mem_A_may, '.', label='Mayores a 0.1')
    ax.plot(ejes_x_A_men[i], mem_A_men, '.', label='Menores a 0.1')
    ax.set_xlabel(labels_x[i], fontsize=15, color='black')
    if i%2==0:
        ax.set_ylabel('Memoria A', fontsize=15, color='black')
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)
    ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Figuras/doblenegcorrmemApos.pdf', dpi=300)

#Para la memoria en B
fig, axs = plt.subplots(2, 2, sharey=True, figsize=(7, 7))
for i, ax in enumerate(axs.flatten()):
    ax.plot(ejes_x_B_may[i], mem_B_may, '.', label='Mayores a 0.1')
    ax.plot(ejes_x_B_men[i], mem_B_men, '.', label='Menores a 0.1')
    ax.set_xlabel(labels_x[i], fontsize=15, color='black')
    if i%2==0:
        ax.set_ylabel('Memoria B', fontsize=15, color='black')
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)
    ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Figuras/doblenegcorrmemBpos.pdf', dpi=300)

#%%
#Calculo correlación ignorando los casos de memoria 0. Uso solo el coeficiente de Spearman porque tiene más sentido por lo que contó Fede
for par_A, par_B, nombre in zip(ejes_x_A_may, ejes_x_B_may, labels_x):
    r_spearman_A, pv_spearman_A = spearmanr(par_A, mem_A_may)
    r_spearman_B, pv_spearman_B = spearmanr(par_B, mem_B_may)

    print(f'Para la memoria en A, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_A} con un p-valor de {pv_spearman_A}.')
    print(f'Para la memoria en B, el coeficiente de Spearman entre memoria y {nombre} es {r_spearman_B} con un p-valor de {pv_spearman_B}.')

#%%
#Figura de correlación entre las memorias mayores de ambas variables y los valores de altos on y off
#Altos On
plt.figure()
plt.plot(ejes_x_A_may[-2], mem_A_may, '.', label='$A$')
plt.plot(ejes_x_B_may[-2], mem_B_may, '.', label='$B$')
plt.xlabel(labels_x[-2], color='black', fontsize=15)
plt.ylabel('Memoria', color='black', fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figuras/doblenegaltosonpos.pdf', dpi=300)

#Altos Off
plt.figure()
plt.plot(ejes_x_A_may[-1], mem_A_may, '.', label='$A$')
plt.plot(ejes_x_B_may[-1], mem_B_may, '.', label='$B$')
plt.xlabel(labels_x[-1], color='black', fontsize=15)
plt.ylabel('Memoria', color='black', fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figuras/doblenegaltosoffpos.pdf', dpi=300)

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

#Anoto los pares que quiero para guardar esas figuras
pares_fig = [('k_ba', 'k_sb'), ('k_ab', 'k_sb'), ('k_ba', 'k_ab'), ('k_ab', 'k_sa'), ('k_ba', 'k_sa'), ('K_sa', 'k_sa')]

pares_en_latex = {
    'k_ba': '$k_{BA}$',
    'k_sb': '$k_{SB}$',
    'k_ab': '$k_{AB}$',
    'k_sb': '$k_{SB}$',
    'k_ba': '$k_{BA}$',
    'k_ab': '$k_{AB}$',
    'k_sa': '$k_{SA}$',
    'k_ba': '$k_{BA}$',
    'k_sa': '$k_{SA}$',
    'K_sa': '$K_{SA}$',
    'k_sa': '$k_{SA}$'
}

#Hago todas las combinaciones posibles
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:-3]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

for i, par, par_num in zip(range(len(pares)), pares, pares_num):
    if par in pares_fig:
        plt.figure()
        #Scatter para hacer el plot XYZ de a pares de parametros con la memoria respectiva
        plt.scatter(*par_num, c=mem_A_ok, marker="o", cmap=cmap1, edgecolor='k', linewidths=0.3)
        cb = plt.colorbar()
        for t in cb.ax.get_yticklabels():
             t.set_fontsize(20)
        plt.xlabel(pares_en_latex[par[0]], fontsize=15, color='black')
        plt.ylabel(pares_en_latex[par[1]], fontsize=15, color='black')
        plt.yticks(fontsize=15, color='black')
        plt.xticks(fontsize=15, color='black')
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'Figuras/comparacion_dobleneg_{par[0]}vs{par[1]}.pdf', dpi=300)

#%%
#Histogramas de los parametros
n_parametros = 8
n_barrido = 10000
parame = lhs.lhs(n_parametros, n_barrido)

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

#%%
#Ahora sí, los gráficos. En escala log-log
for col, param in zip(df, parametros_todos):
    if col in df.columns.to_numpy()[8:]: continue
    if str(col) == 'k_sa':
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        parametro = df.loc[:, col].to_numpy()

        hist, bins = np.histogram(param, bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        # plt.hist(param, bins=logbins, edgecolor="black", label='Todos')
        hist, bins = np.histogram(param, bins=logbins)
        plt.step(bins[:-1], hist, where='post', label='Todos')

        hist, bins = np.histogram(parametro, bins=50)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        # plt.hist(parametro, bins=logbins, edgecolor="black", label='Sistemas Biestables')
        hist, bins = np.histogram(parametro, bins=logbins)
        plt.step(bins[:-1], hist, where='post', label='Sistemas Biestables')

        plt.yticks(fontsize=15, color='black')
        plt.xticks(fontsize=15, color='black')
        plt.xlabel(r"$k_{SA}$", fontsize=15, color='black')
        plt.ylabel("Cantidad de Sistemas", fontsize=15, color='black')
        plt.legend(fontsize=15)
        plt.grid(1)
        plt.tight_layout()
        plt.savefig('Figuras/dobleneghistejemploparam.pdf', dpi=300)

#%%
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
        # plt.hist(parametro, bins=logbins, edgecolor="black", alpha=0.4, label=f"{col}")
        hist, bins = np.histogram(parametro, bins=logbins)
        plt.step(bins[:-1], hist, where='post', label=f"{col}")

plt.ylabel("Cantidad de Sistemas", fontsize=15, color='black')
plt.legend(fontsize=15)
plt.grid(1)
plt.tight_layout()
# plt.savefig('Figuras/histkchicospos.pdf', dpi=300)

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
        # plt.hist(parametro, bins=logbins, edgecolor="black", alpha=0.4, label=f"{col}")
        hist, bins = np.histogram(parametro, bins=logbins)
        plt.step(bins[:-1], hist, where='post', label=f"{col}")

plt.ylabel("Cantidad de Sistemas", fontsize=15, color='black')
plt.legend(fontsize=15)
plt.grid(1)
plt.tight_layout()
# plt.savefig('Figuras/histKgrandespos.pdf', dpi=300)

#%%
#Hago los mismos histogramas separando los k chicos de los K grandes pero en figuras distintas
k_chicos = ['k_ba', 'k_ab', 'k_sa', 'k_sb']
K_grandes =['K_sa', 'K_sb', 'K_ba', 'K_ab']

k_chicos_latex = [r'$k_{BA}$', r'$k_{AB}$', r'$k_{SA}$', r'$k_{SB}$']
K_grandes_latex =[r'$K_{SA}$', r'$K_{SB}$', r'$K_{BA}$', r'$K_{AB}$']

#k chicos
fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 7))
# plt.xscale('log')
for i, ax in enumerate(axs.flatten()):
    col = str(k_chicos[i])
    parametro = df.loc[:, col].to_numpy()
    
    hist, bins = np.histogram(parametro, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    
    hist, bins = np.histogram(parametro, bins=logbins)
    ax.step(bins[:-1], hist, where='post', color='#E24A33')

    ax.hist(parametro, bins=logbins, alpha=0.3, color='#E24A33')

    if i%2==0:
        ax.set_ylabel('Cantidad de Sistemas', fontsize=15, color='black')

    ax.set_xlabel(f"{k_chicos_latex[i]}", fontsize=15, color='black')
    ax.set_xscale('log')
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)

plt.tight_layout()
# plt.savefig('Figuras/dobleneghistkchicos.pdf', dpi=300)

#K grandes
fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 7))
# plt.xscale('log')
for i, ax in enumerate(axs.flatten()):
    col = str(K_grandes[i])
    parametro = df.loc[:, col].to_numpy()
    
    hist, bins = np.histogram(parametro, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    
    hist, bins = np.histogram(parametro, bins=logbins)
    ax.step(bins[:-1], hist, where='post', color='#E24A33')

    ax.hist(parametro, bins=logbins, alpha=0.3, color='#E24A33')

    if i%2==0:
        ax.set_ylabel('Cantidad de Sistemas', fontsize=15, color='black')

    ax.set_xlabel(f"{K_grandes_latex[i]}", fontsize=15, color='black')
    ax.set_xscale('log')
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)

plt.tight_layout()
# plt.savefig('Figuras/dobleneghistKgrandes.pdf', dpi=300)
