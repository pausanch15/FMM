#Este código toma los pickles que genera el V6 y guarda todas esas listas en una base de datos y posteriormente en un csv.

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
plt.ion()

#%%
#Me fijo si el csv existe. Si no existe, lo creo. Si existe, lo levanto y mido la memoria en los sistemas
fname = '2022_02_14-parametros_biestables-modelo_3.csv'

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
        if n%10 == 0:
            params = df.loc[areas[n], :].to_numpy()[:-5]
            A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
        
            plt.figure()
            plt.plot(S, A_s, 'o', label='A')
            plt.plot(S, B_s, '.', label='B')
            plt.grid()
            plt.legend()
            plt.title(f'Área Biestable: {area}')
            plt.show()

            #Calculo memoria
            S_on = df.loc[areas[n], :].to_numpy()[-2]
            S_off = df.loc[areas[n], :].to_numpy()[-1]
            S_bajo = (S_on + S_off)/2
            
            # mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=S_bajo, plot_estimulo=False, plot_memoria=False)
            mem_A[n], mem_B[n] = mm.mide_memoria(*params, S_alto=2, S_bajo=S_bajo, plot_estimulo=True, plot_memoria=True)

else:
    #Traigo todos los archivos pickle que haya en el directorio
    archivos_pickle = glob("*pkl")

    #Guardo las listas que estan en los pickles en un csv
    df = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On', 'S_on', 'S_off'])

    for archivo in archivos_pickle:
        with open(archivo, 'rb') as f:
            resultados = pickle.load(f)

            area = resultados[0]
            param_ancho_alto_s= resultados[1:]
            del(resultados)

            df1 = pd.DataFrame(columns=['K_sa', 'K_sb', 'k_ba', 'k_ab', 'K_ba', 'K_ab', 'k_sa', 'k_sb', 'Ancho', 'Alto Off', 'Alto On', 'S_on', 'S_off'], index=[str(area)])
            df1.loc[str(area), :] = param_ancho_alto_s

            df = df.append(df1)

            del(df1, area, param_ancho_alto_s)

    df.to_csv(fname)
