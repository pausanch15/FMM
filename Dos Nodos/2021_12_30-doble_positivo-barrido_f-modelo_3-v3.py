#Hace lo mismo que el v1 pero se fija si en la carpeta actual encuentra el csv que quiero crear. Si lo encuentra, abre el que encuentra como una base de datos. Si nolo encuentra, lo genera.

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
import latin_hypercube_sampling as lhs
import pandas as pd
from tqdm import tqdm
import os
plt.ion()

#%%
fname = '2022_01_15-parametros_biestables-modelo_3-v2.csv'

if fname in os.listdir():
    #Levanto el csv y veo algunos plots
    df = pd.read_csv(fname, index_col=0)
    
    #Armo un array con todas las áreas biestables
    areas = df.index.to_numpy()
    
    #Por ahora solo encontré dos sistemas biestables, así que los ploteo a ambos
    tiempo_max = 100
    S_max = 1
    S_min = 0
    pasos = 1000
    
    for n, area in enumerate(areas):
        params = df.loc[areas[n], :].to_numpy()[:-3]
        A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
    
        # plt.figure()
        # plt.plot(S, A_s, 'o', label='A')
        # plt.plot(S, B_s, '.', label='B')
        # plt.grid()
        # plt.legend()
        # plt.title(f'Área Biestable: {area}')
        # plt.show()

else:
    #Acá es donde voy a hacer el barrido eligiendo los parámetros con el lhs para la nueva forma de filtrar
    df = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On'])

    n_parametros = 8
    n_barrido = 100
    parametros = lhs.lhs(n_parametros, n_barrido)

    tiempo_max = 100
    S_max = 1
    S_min = 0
    pasos = 1000

    areas = []

    # for params, i in zip(parametros, tqdm(range(n_barrido))):
    for params in tqdm(parametros):
        A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
        resultado_medicion = mb.mide_biestabilidad(A_s, S)
        
        if resultado_medicion is not None:
            area, ancho, alto_off, alto_on = resultado_medicion
            
            df1 = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On'], index=[str(area)])
            df1.loc[str(area), :] = np.array(list(params) + [ancho, alto_off, alto_on])
            df = df.append(df1)
            
            del(df1)
            del(A_s, B_s, S, area, ancho, alto_off, alto_on, resultado_medicion)
        
    df.to_csv(fname)
