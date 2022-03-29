#Acá tomo algunos de los casos que la memoria da menor a 0.1 en los que el codigo que calcula la memoria no funciona, y trato cada caso por separado.

#Librerías 
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
import latin_hypercube_sampling as lhs
import pandas as pd
import mide_memoria_modelo_3 as mm
import os
import pickle
from multiprocessing import Pool
plt.ion()

#Traigo el csv con todos los modelos y los parámetros
fname = '2022_03_07-parametros_biestables-modelo_3.csv'
df = pd.read_csv(fname, index_col=0)

areas = df.index.to_numpy()
anchos = df.loc[:, 'Ancho'].to_numpy()
altos_on = df.loc[:, 'Alto On'].to_numpy()
altos_off = df.loc[:, 'Alto Off'].to_numpy()
S_on = df.loc[:, 'S_on'].to_numpy()
S_off = df.loc[:, 'S_off'].to_numpy()

#Levanto la memoria de los sistemas, calculada como hasta ahora
with open(f'2022_03_07-mem_A.pkl', 'rb') as f:
            mem_A = pickle.load(f)                
with open(f'2022_03_07-mem_B.pkl', 'rb') as f:
            mem_B = pickle.load(f)

#Saco los valores que dan memoria infinita
i_mem_A = np.where(mem_A>1e3)
i_mem_B = np.where(mem_B>1e3)

areas = np.delete(areas, i_mem_B[0])
anchos = np.delete(anchos, i_mem_B[0])
altos_on = np.delete(altos_on, i_mem_B[0])
altos_off = np.delete(altos_off, i_mem_B[0])
mem_A = np.delete(mem_A, i_mem_B[0])
mem_B = np.delete(mem_B, i_mem_B[0])

#Me quedo con los indices de los valores de memoria menores a 0.1
i_mem_A_men = np.where(mem_A<0.1)

# Tomo alguno de los sistemas con la memoria mal calculada y la recalculo a mano
#Las curvas de histéresis de los casos n = 0, 1, 2, 3 no son muy bonitas, así que arranco con n = 4
n = i_mem_A_men[0][4]
params = df.loc[areas[n], :].to_numpy()[:-5]

#El modelo que se va a integrar con estímulo y sin él
#wos = without stimulus
#ws = with stimulus
def modelo_wos(vars,params):
    
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
    dA = S*k_sa*(1-A)/(K_sa+1-A) + B*k_ba*(1-A)/(K_ba+1-A) - k_ba*A/(K_ba+A)
    dB = S*k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

def modelo_ws(vars, params, interpolar_estimulo, tiempo):
    
    S = interpolar_estimulo(tiempo) #aca lo interpola

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
    dA = S*k_sa*(1-A)/(K_sa+1-A) + B*k_ba*(1-A)/(K_ba+1-A) - k_ba*A/(K_ba+A)
    dB = S*k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

#Integro sin estímulo
tiempo_min = 100
tiempo_max = 1000
S = S_bajo

params = np.insert(params, 0, S) #Agrego S a params

condiciones_iniciales = [0, 0]
tiempo_wos, variables_wos = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)

A_wos = variables_wos[0]
B_wos = variables_wos[1]

#Integro con estímulo  
#Definimos el estimulo variable
N_estimulo = 10000 #resolucion para el estimulo
tiempo_max_estimulo = 1000 #tiempo hasta donde llega
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
tiempo_subida = tiempo_wos[-1] + 10
tiempo_bajada = tiempo_subida + 50
#Estimulo, multiplico un vector de unos por uno de ceros y unos (falses y trues de la desigualdad) que me da el escalón
estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada) 
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))
    
condiciones_iniciales = [0,0]
tiempo_min = tiempo_bajada + 20
tiempo_max = 1000

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo_ws, variables_ws = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

A_ws = variables_ws[0]
B_ws = variables_ws[1]

#Grafico los sistemas y el escalon si plot_estimulo es True
if plot_estimulo == True:
    plt.figure()
    plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo_ws[-1]], estimulo[tiempo_estimulo<tiempo_ws[-1]], label='Estímulo', color='k')
    plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()

if plot_memoria == True:
    plt.figure()
    plt.plot(tiempo_ws, A_ws, label='A', color='c')
    plt.plot(tiempo_ws, B_ws, label='B', color='g')
    plt.xlabel('Tiempo'); plt.ylabel('A, B'); plt.legend(); plt.grid()

#Calculo la memoria
memoria_A = A_ws[-1] - A_wos[-1]
memoria_B = B_ws[-1] - B_wos[-1]
