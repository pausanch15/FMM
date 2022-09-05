#Hago el gráfico de barras que propuso Fede para comparar los coeficientes de Spearman y los p-valores

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
#Armo listas con los valores que quiero poner en los gráficos de barras
#Levanto el csv del Doble Positivo
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

#Calculo correlación ignorando los casos de memoria 0. Uso solo el coeficiente de Spearman porque tiene más sentido por lo que contó Fede
dp_A_rs_spearman = []
dp_B_rs_spearman = []

dp_A_pvs = []
dp_B_pvs = []

for par_A, par_B, nombre in zip(ejes_x_A_may, ejes_x_B_may, labels_x):
    r_spearman_A, pv_spearman_A = spearmanr(par_A, mem_A_may)
    r_spearman_B, pv_spearman_B = spearmanr(par_B, mem_B_may)

    dp_A_rs_spearman.append(r_spearman_A); dp_B_rs_spearman.append(r_spearman_B); dp_A_pvs.append(pv_spearman_A); dp_B_pvs.append(pv_spearman_B)

#Levanto el csv del Doble Negativo
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
        
#Calculo correlación ignorando los casos de memoria 0. Uso solo el coeficiente de Spearman porque tiene más sentido por lo que contó Fede
dn_A_rs_spearman = []
dn_B_rs_spearman = []

dn_A_pvs = []
dn_B_pvs = []

for par_A, par_B, nombre in zip(ejes_x_A_may, ejes_x_B_may, labels_x):
    r_spearman_A, pv_spearman_A = spearmanr(par_A, mem_A_may)
    r_spearman_B, pv_spearman_B = spearmanr(par_B, mem_B_may)

    dn_A_rs_spearman.append(r_spearman_A); dn_B_rs_spearman.append(r_spearman_B); dn_A_pvs.append(pv_spearman_A); dn_B_pvs.append(pv_spearman_B)

#%%
#Armo el gráfico de barras
labels_x = ['Área', 'Ancho', 'Alto On', 'Altos Off']
labels_y = ['rSpearman', 'P-Valor']
scales = ['linear', 'log']

A = [[dp_A_rs_spearman, dn_A_rs_spearman], [dp_A_pvs, dn_A_pvs]]
B = [[dp_B_rs_spearman, dn_B_rs_spearman], [dp_B_pvs, dn_B_pvs]]

#Para la memoria en A
fig, axs = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(15, 5))
for i, ax in enumerate(axs.flatten()):
    # ax.bar(labels_x, A[i][0], label='Doble Positivo', width=0.4)
    # ax.bar(labels_x, A[i][1], label='Doble Negativo', width=0.4)
    yaxis = np.arange(len(labels_x))
    height=0.4
    ax.barh(yaxis+height/2, A[i][0], label='Doble Positivo', height=height)
    ax.barh(yaxis-height/2, A[i][1], label='Doble Negativo', height=height)
    ax.set_xlabel(labels_y[i], color='black', fontsize=15)
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)
    ax.legend(fontsize=15)
    ax.set_yticks(yaxis, labels_x)
    ax.set_xscale(scales[i])
plt.tight_layout()
plt.savefig('Figuras/com_corr_memA.pdf', dpi=300)

#Para la memoria en B
fig, axs = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(15, 5))
for i, ax in enumerate(axs.flatten()):
    yaxis = np.arange(len(labels_x))
    height=0.4
    ax.barh(yaxis+height/2, B[i][0], label='Doble Positivo', height=0.4)
    ax.barh(yaxis-height/2, B[i][1], label='Doble Negativo', height=0.4)
    ax.set_xlabel(labels_y[i], color='black', fontsize=15)
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)
    ax.legend(fontsize=15)
    ax.set_yticks(yaxis, labels_x)
    ax.set_xscale(scales[i])
plt.tight_layout()
plt.savefig('Figuras/com_corr_memB.pdf', dpi=300)
