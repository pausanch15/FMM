#Acá genero los csv para el modelo 2
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import pandas as pd
plt.ion()

#%%
#Defino el modelo
def modelo(vars, params):

    S = params[0]  
    
    # Parámetros de síntesis de A y B
    k_sa = params[1]
    k_sb = params[2]
    K_sa = params[3]
    K_sb = params[4]

    # Parámetros de inhibición mutua
    k_ba = params[5]
    k_ab = params[6]
    K_ba = params[7]
    K_ab = params[8]

    # Variables
    A=vars[0]
    B=vars[1]

    # Sistema de ecuaciones
    dA = S*k_sa*(1-A)/(K_sa + 1-A) - k_ba*A/((K_ba+A)*(B+1))
    dB = k_sb*(1-B)/(K_sb + 1-B) - k_ab*B/((K_ab+B)*(A+1))
    
    return np.array([dA,dB])

#%%
#Preparo los parámetros para empezar a buscar la zona biestable
#Primero voy a barrer solamente en los parámetros de feedback: k_sa y k_sb
condiciones_iniciales = [0, 0]

K_sa = 1
K_sb = 1

k_ba = 1
k_ab = 1
K_ba = 0.01
K_ab = 0.01

k_sa = 2
k_sb_s = np.linspace(0, 3, 8)

tiempo_max = 1000
S_max = 2 #El máximo input que voy a usar
pasos = 50

#Armo el dataframe vacío
S_ida = np.linspace(0, S_max, pasos)
S_vuelta = np.linspace(S_max, 0, pasos)
S = np.concatenate((S_ida, S_vuelta))

columnas = [f'k_sb={ksb:.3f}' for ksb in k_sb_s]
codes_0 = sorted([i for i in range(len(k_sb_s))]*2)
codes_1 = [0, 1]*len(k_sb_s)

cols = pd.MultiIndex(levels=[columnas, ["A", "B"]], codes=[codes_0, codes_1])

df = pd.DataFrame(columns=cols, index=S)

#Empiezo el recorrido
for ksb, col in zip(k_sb_s, columnas):
    lista_condiciones_iniciales = [[0, 0]]
    k_sb = ksb
    A_conv = []
    B_conv = []

    #Ida
    for i, s in enumerate(S_ida):
        condiciones_iniciales = lista_condiciones_iniciales[-1]
        params = [s, k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_conv.append(variables[0][-1])
        B_conv.append(variables[1][-1])
        lista_condiciones_iniciales.append([variables[0][-1], variables[1][-1]])
        del(tiempo, variables)

    #Vuelta
    for i, s in enumerate(S_vuelta):
        condiciones_iniciales = lista_condiciones_iniciales[-1]
        params = [s, k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_conv.append(variables[0][-1])
        B_conv.append(variables[1][-1])
        lista_condiciones_iniciales.append([variables[0][-1], variables[1][-1]])
        del(tiempo, variables)

    #Guardo en df
    df.loc[:, (col, 'A')] = A_conv
    df.loc[:, (col, 'B')] = B_conv    
    del(A_conv, B_conv, lista_condiciones_iniciales)

#Guardo el dataframe como csv
#El nombre es 2021_11_16-doble_positivo-modelo_1-ksa_valorka_ksb_kbminimo_kbmaximo
# df.to_csv('2021_11_18-doble_positivo-modelo_1-ksa_2-ksb_4_6.csv')
