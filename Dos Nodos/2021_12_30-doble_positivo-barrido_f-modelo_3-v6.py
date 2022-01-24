#Pruebo hacer lo mismo que vengo haciendo hasat ahora pero de manera que corra para varios sets de parámetros en paralelo.

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
import time
import pickle
from multiprocessing import Pool
plt.ion()

#%%
fname = '2022_01_24-parametros_biestables-modelo_3.csv'

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

    #Defino la función que integra, se fija si los sistemas son biestables, y si lo son, se guarda la lista de parámetros y características en un pickle
    def integra(params):
        #i tiene que ser el indice
        #params tiene que ser la lista de parámetros
        i = params[1]
        params = params[0][i]

        #Checkeo
        print(i)
        print()
        
        #El nombre del pkl que va guardar los parámetros en caso de ser biestables
        nombre_archivo = f'parametros_{i}'

        #Efectivamente integramos
        A_s, B_s, S = gucd.gucd_modelo_3(*params, tiempo_max, S_max, S_min, pasos)
        resultado_medicion = mb.mide_biestabilidad(A_s, S)

        #Nos fijamos si es biestable el sistema
        if resultado_medicion is not None:
            area, ancho, alto_off, alto_on = resultado_medicion
            lista_resultado = [area] + params + [ancho, alto_off, alto_on]

            #Si es biestable, nos guardamos lista_resultado en un pickle
            with open(f'{nombre_archivo}.pkl', 'wb') as f:
                pickle.dump(lista_resultado)

    #Defino la lista de inputs que le voy a pasar a la función integra
    lista_inputs = []
    for i in range(n_barrido):
        # inp = parametros + [i]
        inp = [parametros, i, n_barrido]
        lista_inputs.append(inp)
        del(inp)

    #Corro de forma paralela
    if __name__ == '__main__':
        p = Pool()
        p.map(integra, lista_inputs)
