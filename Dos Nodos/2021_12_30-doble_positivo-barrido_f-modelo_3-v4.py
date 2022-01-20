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
#También veo los sistemas con los que trabajo
tiempo_max = 100
S_max = 1
S_min = 0
pasos = 1000

#Arrays donde me voy a guardar los valores de memoria
mem_A = np.zeros_like(areas)
mem_B = np.zeros_like(areas)

for n, area in enumerate(areas):
    parametros = df.loc[areas[n], :].to_numpy()[:8]

    # Veo el going up coming down de estos sistemas
    # A_s, B_s, S = gucd.gucd_modelo_3(*parametros, tiempo_max, S_max, S_min, pasos)
# 
    # plt.figure()
    # plt.plot(S, A_s, 'o', label='A')
    # plt.plot(S, B_s, '.', label='B')
    # plt.grid()
    # plt.legend()
    # plt.title(f'Área Biestable: {area}')
    # plt.show()

    #Calculo memoria
    mem_A[n], mem_B[n] = mm.mide_memoria(*parametros, S_alto=2, S_bajo=0.5, plot_estimulo=False, plot_memoria=False)
    # mem_A[n], mem_B[n] = mm.mide_memoria(*parametros, S_alto=2, S_bajo=0.5, plot_estimulo=True, plot_memoria=True)
