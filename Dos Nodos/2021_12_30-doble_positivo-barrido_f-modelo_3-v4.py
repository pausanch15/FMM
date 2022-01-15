#Levanto el csv que obtuve antes y mido memoria en esos modelos y en el que pasó Fede.

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
from scipy import interpolate
import runge_kuta_estimulo as rks
import mide_memoria_modelo_3 as mm
plt.ion()

#%%
#Levanto el csv que ya existe
fname = '2022_01_13-parametros_biestables-modelo_3.csv'
df = pd.read_csv(fname, index_col=0)

#Le agrego los parámetros que pasó Fede y lo que dio medir area, alto y ancho con ellos
K_sa = .1
K_sb = .1
k_ba = 1
k_ab = 1
K_ba = .1
K_ab = .1
k_sa = 1
k_sb = 1

tiempo_max = 100
S_max = 1
S_min = 0
pasos = 1000

A_s, B_s, S = gucd.gucd_modelo_3(K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb, tiempo_max, S_max, S_min, pasos)

area, ancho, alto_off, alto_on = mb.mide_biestabilidad(A_s, S)

df1 = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On'], index=[str(area)])
df1.loc[str(area), :] = np.array([K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb] + [ancho, alto_off, alto_on])
df = df.append(df1)
del(df1)
del(A_s, B_s, S, area, ancho, alto_off, alto_on)

#Me quedo las áreas
areas = df.index.to_numpy()

#Intento medir memoria en estos sistemas
#Para eso tengo que pasarle un estímulo escalón al sistema ya estabilizado
# memorias_A = []
# memorias_B = []
for n, area in enumerate(areas):
    parametros = df.loc[areas[n], :].to_numpy()[:8]
    mem_A, mem_B, t = mm.mide_memoria(parametros)
    # memorias_A.append(mem_A)
    # memorias_B.append(mem_B)

    #Prueboo los distintos contadores de memoria que propuso Fede
    #OJO: el dice que pruebe estas cosas una vez que el sistema ya estabilizó habiendo sacado el escalón, pero yo los voy a probar un en el transitorio
    
    #Ploteo
    plt.figure()
    plt.title(f'Conjunto de parámetros {n}')
    
    plt.plot(t, mem_A, label='Memoria A', color='red', linestyle='solid')
    plt.plot(t, mem_B, label='Memoria B', color='red', linestyle='dashed')

    plt.plot(t, np.abs(mem_A), label='Módulo Memoria A', color='blue', linestyle='solid')
    plt.plot(t, np.abs(mem_B), label='Módulo Memoria B', color='blue', linestyle='dashed')

    plt.plot(t, mem_A**2, label='Cuadrado Memoria A', color='green', linestyle='solid')
    plt.plot(t, mem_B**2, label='Cuadrado Memoria B', color='green', linestyle='dashed')
    
    plt.show(), plt.grid(), plt.legend()
