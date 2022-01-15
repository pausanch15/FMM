#Función que hace el going up comming down (gucd) para el modelo 3 del doble positivo
#La idea es que la función hace lo que hace para los parámetros que le pases
#Para el Modelo 3, S no puede arrancar siendo exactamente 0

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v2 as mb
import runge_kuta as rk
plt.ion()

#%%
def gucd_modelo_3(K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb, tiempo_max, S_max, S_min, pasos):
    '''
    Las condiciones iniciales que usa son A=0 y B=0.
    '''
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
        dA = S*k_sa*(1-A)/(K_sa+1-A) + B*k_ba*(1-A)/(K_ba+1-A) - k_ba*A/(K_ba+A)
        dB = S*k_sb*(1-B)/(K_sb+1-B) + A*k_ab*(1-B)/(K_ab+1-B) - k_ab*B/(K_ab+B)
        
        return np.array([dA,dB])
    
    A_s = []
    B_s = []
    
    lista_condiciones_iniciales = [[0, 0]] 
    
    #Ida
    S_ida = np.linspace(S_min, S_max, pasos) #Los inputs que voy barriendo
    for i, s in enumerate(S_ida):
        condiciones_iniciales = lista_condiciones_iniciales[-1]
        params = [s, k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_s.append(variables[0][-1])
        B_s.append(variables[1][-1])
        lista_condiciones_iniciales.append([variables[0][-1], variables[1][-1]])
        del(tiempo, variables)
    
    #Vuelta
    S_vuelta = np.linspace(S_max, S_min, pasos) #Los inputs que voy barriendo: los de antes pero al revés
    for i, s in enumerate(S_vuelta):
        condiciones_iniciales = lista_condiciones_iniciales[-1]
        params = [s, k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_s.append(variables[0][-1])
        B_s.append(variables[1][-1])
        lista_condiciones_iniciales.append([variables[0][-1], variables[1][-1]])
        del(tiempo, variables)
    
    #Junto los inputs de ida y vuelta
    S = np.concatenate((S_ida, S_vuelta))

    #Devuelvo cosas
    return A_s, B_s, S
