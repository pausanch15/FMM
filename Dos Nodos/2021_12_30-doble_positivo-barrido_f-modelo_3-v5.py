#Hace lo mismo que el v3, pero voy a probar rescalar los valores que pueden tomar los parámetros mediante el lhs

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
import latin_hypercube_sampling as lhs
import pandas as pd
from tqdm import tqdm
import mide_memoria_modelo_3 as mm
import os
plt.ion()

#%%
fname = '2022_01_20-parametros_biestables-modelo_3.csv'

if fname in os.listdir():
    #Levanto el csv y veo algunos plots
    df = pd.read_csv(fname, index_col=0)
    
    #Armo un array con todas las áreas biestables
    areas = df.index.to_numpy()

    #Arrays donde me voy a guardar los valores de memoria
    mem_A = np.zeros_like(areas)
    mem_B = np.zeros_like(areas)
    
    #Por ahora solo encontré un sistema biestable
    tiempo_max = 100
    S_max = 1
    S_min = 0
    pasos = 1000
    
    for n, area in enumerate(areas):
        params = df.loc[areas[n], :].to_numpy()[:-3]
        A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
    
        plt.figure()
        plt.plot(S, A_s, 'o', label='A')
        plt.plot(S, B_s, '.', label='B')
        plt.grid()
        plt.legend()
        plt.title(f'Área Biestable: {area}')
        plt.show()

        #Calculo memoria
        # mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=0.8, plot_estimulo=False, plot_memoria=False)
        mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=0.8, plot_estimulo=True, plot_memoria=True)

else:
    #Acá es donde voy a hacer el barrido eligiendo los parámetros con el lhs para la nueva forma de filtrar
    df = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On'])

    n_parametros = 8
    n_barrido = 100
    parame = lhs.lhs(n_parametros, n_barrido)

    #Elegimos entre que valores queremos la distribucion
    #Estos van a ir entre 10**(-2) y 10**2 con distribucion log-uniforme
    parametros= np.copy(parame)

    #Permito solo que los k chiquitos se muevan entre 10**(-2) y 10**2, los K grandes los dejo entre 0 y 1
    parametros[:, 2] = 10**((parame[:,2]-0.5)*4)
    parametros[:, 3] = 10**((parame[:,3]-0.5)*4)
    parametros[:, 6] = 10**((parame[:,6]-0.5)*4)
    parametros[:, 7] = 10**((parame[:,7]-0.5)*4)

    #Defino lo necesario para integrar
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
