#Esta función mide memoria de la forma más básica para el modelo 3. Después con lo que esto devuelve se pueden probar los otros contadores de memoria

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

#%% 
#Defino el modelo con y sin el estimulo
def modelo_wos(varis,params):
    
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
    A=varis[0]
    B=varis[1]

    # Sistema de ecuaciones
    dA = S*k_sa*(1-A)/(K_sa+1-A) + B*k_ba*(1-A)/(K_ba+1-A) - k_ba*A/(K_ba+A)
    dB =   k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

def modelo_ws(varis, params, interpolar_estimulo, tiempo):
    
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
    A=varis[0]
    B=varis[1]

    # Sistema de ecuaciones
    dA = S*k_sa*(1-A)/(K_sa+1-A) + B*k_ba*(1-A)/(K_ba+1-A) - k_ba*A/(K_ba+A)
    dB =   k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])
#%% 
#Armo el estímulo
def armar_estimulo_pulso(t_subida, ancho, t_max, N_puntos, S_alto, S_bajo, plot=False):
    t_estimulo = np.linspace(0,t_max,N_puntos) #vector de tiempo para el estimulo
    t_bajada = t_subida + ancho
    #Estimulo, multiplico un vector de unos por uno de ceros y unos (falses y trues de la desigualdad) que me da el escalón
    estimulo = S_alto*np.ones(N_puntos)*(t_estimulo>t_subida)*(t_estimulo<t_bajada)
    estimulo = estimulo + S_bajo*np.ones(N_puntos)*((t_estimulo<t_subida)+(t_estimulo>t_bajada))
    estimulo[0] = S_bajo
    
    if plot:
        plt.figure()
        plt.title('Estímulo')
        plt.plot(t_estimulo, estimulo)
        plt.show()
        
    return t_estimulo,estimulo
#%%
#La función
# def mide_memoria(params, S_alto=100, S_bajo=0.1):
def mide_memoria(K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb, S_alto, S_bajo, plot_mem=False, plot_est=False):
    '''
    parametros: lista de parámetros del modelo SIN el S
    S_alto y S_bajo son esos valores para el escalón
    '''
    #El modelo que se va a integrar con estímulo y sin él
    #wos = without stimulus
    #ws = with stimulus

    #Construyo a lista de parámetros
    params = [k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
    
    #Integro sin estímulo
    tiempo_min = 1000
    tiempo_max = 1000
    
    params = np.insert(params, 0, S_bajo) #Agrego S_bajo a params
    condiciones_iniciales = [0, 0]
    tiempo_wos, variables_wos = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)
    A_wos = variables_wos[0]
    B_wos = variables_wos[1]

    #Integro con estímulo
    #Definimos el estimulo variable
    N_puntos = 10000 #resolucion para el estimulo
    t_max = 2000 #tiempo hasta donde llega
    t_subida = 0
    ancho = 50
    t_estimulo,estimulo = armar_estimulo_pulso(t_subida, ancho, t_max, N_puntos, S_alto, S_bajo, plot=plot_est)
    interpolar_estimulo = interpolate.interp1d(t_estimulo,estimulo)
    
    condiciones_iniciales = [A_wos[-1], B_wos[-1]]
    tiempo_min = t_subida + ancho + 20
    tiempo_ws, variables_ws = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    
    #Concateno variables wos y ws
    variables_ws = np.concatenate((variables_wos,variables_ws), axis=1)
    A_ws = variables_ws[0]
    B_ws = variables_ws[1]

    #Grafico los sistemas y el escalon si plot_mem=True
    estimulo_bajo = S_bajo*np.ones(len(tiempo_wos))
    estimulo = np.concatenate((estimulo_bajo, interpolar_estimulo(tiempo_ws)))
    tiempo_ws = np.concatenate((tiempo_wos, tiempo_ws+tiempo_wos[-1]))

    if plot_mem:
        plt.figure()
        plt.title('Memoria')
        
        #Ploteo el estímulo
        plt.plot(tiempo_ws, estimulo, label='Estímulo')

        # #Ploteo A_wos
        # plt.plot(tiempo_wos, A_wos, label='Sin Estímulo')
        
        #Ploteo A_ws
        plt.plot(tiempo_ws, A_ws, label='Con Estímulo')

        plt.legend()
        plt.show()

    #Calculo la memoria
    memoria_A = A_ws[-1] - A_wos[-1]
    memoria_B = B_ws[-1] - B_wos[-1]

    return [memoria_A, memoria_B]
