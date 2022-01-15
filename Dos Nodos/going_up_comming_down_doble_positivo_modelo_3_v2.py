#Hace lo mismo que su v1 pero mejorado

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
    
    condiciones_iniciales = [0, 0]

    params = [k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
    
    #Ida
    S_ida = np.linspace(S_min, S_max, pasos) #Los inputs que voy barriendo

    A_B_s = np.zeros((pasos*2, 2)) #Matriz donde guardo los resultados
    
    for i, s in enumerate(S_ida):
        tiempo, variables = rk.integrar(modelo, [s, *params], condiciones_iniciales, tiempo_max)
        A_B_s[i] = variables[0][-1], variables[1][-1] 
        condiciones_iniciales = A_B_s[i]
        del(tiempo, variables)
    
    #Vuelta
    S_vuelta = np.linspace(S_max, S_min, pasos) #Los inputs que voy barriendo: los de antes pero al revés
    for i, s in enumerate(S_vuelta):
        tiempo, variables = rk.integrar(modelo, [s, *params], condiciones_iniciales, tiempo_max)
        A_B_s[i+pasos] = variables[0][-1], variables[1][-1] 
        condiciones_iniciales = A_B_s[i+pasos]
        del(tiempo, variables)
    
    #Junto los inputs de ida y vuelta
    S = np.concatenate((S_ida, S_vuelta))

    #Devuelvo cosas
    return [*A_B_s.T, S] #Devuelve [A_s, B_s, S]
