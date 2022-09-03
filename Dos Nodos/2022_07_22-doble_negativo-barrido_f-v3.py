#Levanto los pkl que salieron del v2 y armo el csv

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_negativo_v1 as gucd
import pandas as pd
import os
from glob import glob
import pickle
plt.ion()

#%%
#Me fijo si el csv existe. Si no existe, lo creo. Si existe, lo levanto y mido la memoria en los sistemas
fname = '2022_08_31-parametros_biestables-doble_negativo.csv'

if fname in os.listdir():
    #Levanto el csv y veo algunos plots
    df = pd.read_csv(fname, index_col=0)
    
    #Armo un array con todas las áreas biestables
    areas = df.index.to_numpy()

    #Me fijo cuántos sistemas biestables obtuve
    print(f'Hay {len(areas)} sistemas biestables')

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
