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

#La función
def mide_memoria(K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb, S_alto=100, S_bajo=0.1, plot_estimulo=False, plot_memoria=False):
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

    params = [k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
    
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
        fig = plt.figure(figsize=(10, 7))
        plt.subplot(211)
        plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo_ws[-1]], estimulo[tiempo_estimulo<tiempo_ws[-1]], label='Estímulo', color='k')
        plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()
        plt.subplot(212)
        plt.plot(tiempo_ws, A_ws, label='A', color='c')
        plt.plot(tiempo_ws, B_ws, label='B', color='g')
        plt.xlabel('Tiempo'); plt.ylabel('A, B'); plt.legend(); plt.grid(); plt.show()

    #Calculo la memoria
    memoria_A = A_ws[-1] - A_wos[-1]
    memoria_B = B_ws[-1] - B_wos[-1]

    return [memoria_A, memoria_B]
