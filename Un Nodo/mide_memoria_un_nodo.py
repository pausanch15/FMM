#Librerías
import matplotlib.pyplot as plt
import numpy as np
# import mide_biestabilidad_v3 as mb
import runge_kuta as rk
# import going_up_comming_down_doble_positivo_modelo_3_v2 as gucd
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#La función
def mide_memoria(k2, k1=1, K2=0.5, k3=1, n=3, S_alto=3, S_bajo=0.01, plot_estimulo=False, plot_memoria=False):
    '''
    parametros: lista de parámetros del modelo SIN el S
    S_alto y S_bajo son esos valores para el escalón
    '''
    #El modelo que se va a integrar con estímulo y sin él
    #wos = without stimulus
    #ws = with stimulus
    def modelo_wos(vars,params):
        # Parámetros
        S = params[0]
        k1 = params[1]
        k2 = params[2]
        K2 = params[3]
        n = params[4]
        k3 = params[5]
    
        # Variables
        A=vars[0]
    
        # Sistema de ecuaciones
        # estimulo + feedback - inactivacion
        dA = k1*S*(1-A) + k2*(1-A)*A**n/(K2+A)**n - k3*A
        
        return np.array([dA])
    
    def modelo_ws(vars, params, interpolar_estimulo, tiempo):
        
        S = interpolar_estimulo(tiempo) #aca lo interpola
    
        # Parámetros
        # S = params[0]
        k1 = params[1]
        k2 = params[2]
        K2 = params[3]
        n = params[4]
        k3 = params[5]
    
        # Variables
        A=vars[0]
    
        # Sistema de ecuaciones
        # estimulo + feedback - inactivacion
        dA = k1*S*(1-A) + k2*(1-A)*A**n/((K2**n)+(A**n)) - k3*A
        
        return np.array([dA])

    params = [k1, k2, K2, n, k3]
    
    #Integro sin estímulo
    tiempo_min = 100
    tiempo_max = 1000
    S = S_bajo
    
    params = np.insert(params, 0, S) #Agrego S a params
    
    condiciones_iniciales = [0]
    tiempo_wos, variables_wos = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)
    
    A_wos = variables_wos[0]

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
        
    condiciones_iniciales = [0]
    tiempo_min = tiempo_bajada + 20
    tiempo_max = 1000
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo_ws, variables_ws = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    
    A_ws = variables_ws[0]

    #Grafico los sistemas y el escalon si plot_estimulo es True
    if plot_estimulo == True:
        plt.figure()
        plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo_ws[-1]], estimulo[tiempo_estimulo<tiempo_ws[-1]], label='Estímulo', color='k')
        plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()

    if plot_memoria == True:
        plt.figure()
        plt.plot(tiempo_ws, A_ws, label='A', color='c')
        plt.xlabel('Tiempo'); plt.ylabel('A'); plt.legend(); plt.grid()

    #Calculo la memoria
    memoria_A = A_ws[-1] - A_wos[-1]

    return memoria_A
