#Hago pruebas y trabajo sobre la función edl mide_memoria_modelo_3.py
#Librerías
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#La función
def mide_memoria(K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb, S_alto=100, S_bajo=0.1):
    '''
    parametros: lista de parámetros del modelo SIN el S
    S_alto y S_bajo son esos valores para el escalón
    '''
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
        dB = k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
        
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
        dB = k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
        
        return np.array([dA,dB])

    params = [k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
    
    #Integro sin estímulo
    tiempo_min = 1000
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
    tiempo_max_estimulo = 2000 #tiempo hasta donde llega
    tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
    tiempo_subida = 0
    tiempo_bajada = tiempo_subida + 50
    #Estimulo, multiplico un vector de unos por uno de ceros y unos (falses y trues de la desigualdad) que me da el escalón
    estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada)
    estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))
    estimulo[0] = S_bajo
        
    condiciones_iniciales = [A_wos[-1], B_wos[-1]]
    tiempo_min = tiempo_bajada + 20
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo_ws, variables_ws = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    
    variables_ws = np.concatenate((variables_wos,variables_ws), axis=1)
    A_ws = variables_ws[0]
    B_ws = variables_ws[1]

    #Grafico los sistemas y el escalon
    #Ploteo el estímulo
    estimulo_bajo = S_bajo*np.ones(len(tiempo_wos))
    estimulo = np.concatenate((estimulo_bajo, interpolar_estimulo(tiempo_ws)))
    tiempo_ws = np.concatenate((tiempo_wos, tiempo_ws+tiempo_wos[-1]))
    plt.plot(tiempo_ws, estimulo)

    #Ploteo As
    plt.plot(tiempo_ws, A_ws)

    #Calculo la memoria
    memoria_A = A_ws[-1] - A_wos[-1]
    memoria_B = B_ws[-1] - B_wos[-1]

    return [memoria_A, memoria_B]

#%%
#Pruebo la función con el ejemplo que usó Fede
params = np.array([1.20032999e-02, 2.19836442e-01, 1.44079207e-02, 4.13998718e+00, 1.20032999e-02, 1.20032999e-02, 1.34102344e-01, 5.70161534e+01])
condiciones_iniciales = [0, 0]

mem_A, mem_B = mide_memoria(*params, S_alto=10, S_bajo=0.1)

#%%
#Lo pruebo sin la función
params = np.array([1.20032999e-02, 2.19836442e-01, 1.44079207e-02, 4.13998718e+00, 1.20032999e-02, 1.20032999e-02, 1.34102344e-01, 5.70161534e+01])
condiciones_iniciales = [0, 0]
S_alto = 10
S_bajo = 0.1

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
    dB = k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
    
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
    dB = k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

#Integro sin estímulo
tiempo_min = 1000
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
tiempo_max_estimulo = 2000 #tiempo hasta donde llega
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
tiempo_subida = 0
tiempo_bajada = tiempo_subida + 50
#Estimulo, multiplico un vector de unos por uno de ceros y unos (falses y trues de la desigualdad) que me da el escalón
estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada)
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))
estimulo[0] = S_bajo
    
condiciones_iniciales = [A_wos[-1], B_wos[-1]]
tiempo_min = tiempo_bajada + 20

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo_ws, variables_ws = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

variables_ws = np.concatenate((variables_wos,variables_ws), axis=1)
A_ws = variables_ws[0]
B_ws = variables_ws[1]

#Grafico los sistemas y el escalon
plt.figure()

#Ploteo el estímulo
estimulo_bajo = S_bajo*np.ones(len(tiempo_wos))
estimulo = np.concatenate((estimulo_bajo, interpolar_estimulo(tiempo_ws)))
tiempo_ws = np.concatenate((tiempo_wos, tiempo_ws+tiempo_wos[-1]))
plt.plot(tiempo_ws, estimulo)

#Ploteo As
plt.plot(tiempo_ws, A_ws)

#Calculo la memoria
memoria_A = A_ws[-1] - A_wos[-1]
memoria_B = B_ws[-1] - B_wos[-1]


