#Integra el feedback negativo

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#%%
def integra_FN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max):
    '''
    Las condiciones iniciales que usa son X=y=Y=0
    Devuelve (tiempo, variables)
    '''
    #Defino el modelo
    def modelo(vars, params):
        #Parámetros
        C1 = params[0] 
        dX1 = params[1] 
        dYX1 = params[2]
        KYX1 = params[3]
        Ty1 = params[4] 
        dy1 = params[5]
        TY1 = params[6]
        dY1 = params[7]

        # Variables
        X=vars[0]
        y=vars[1]
        Y=vars[2]
    
        # Sistema de ecuaciones
        dX = C1 - dX1*X - dYX1*Y*(X/(KYX1+X))
        dy = Ty1*(X/(1+X)) - dy1*y
        dY = TY1*y - dy1*Y
        
        return np.array([dX, dy, dY])

    #Integramos
    condiciones_iniciales = [0, 0, 0]
    params = [C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1]
    tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)

    return tiempo, variables

def integra_FN_estimulo(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max=1000, S_alto=100, S_bajo=0.1, N_estimulo=10000, tiempo_max_estimulo=1000, tiempo_subida=100, tiempo_bajada=300):
    '''
    Las condiciones iniciales que usa son X=y=Y=0
    Devuelve (tiempo, variables, tiempo_estimulo, estimulo)
    '''
    #Defino el modelo
    def modelo(vars, params, interpolar_estimulo, tiempo):
        #Parámetros
        C1 = params[0] 
        dX1 = params[1] 
        dYX1 = params[2]
        KYX1 = params[3]
        Ty1 = params[4] 
        dy1 = params[5]
        TY1 = params[6]
        dY1 = params[7]

        # Variables
        X=vars[0]
        y=vars[1]
        Y=vars[2]
    
        # Sistema de ecuaciones
        dX = C1 - dX1*X - dYX1*Y*(X/(KYX1+X))
        dy = Ty1*(X/(1+X)) - dy1*y
        dY = TY1*y - dy1*Y
        
        return np.array([dX, dy, dY])

    #Defino el estímulo con caída exponencial
    tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo)
    
    x = np.where(tiempo_estimulo>tiempo_bajada, tiempo_estimulo, np.nan)
    x_inicio_caida = len(np.where(np.isnan(x))[0])
    caida = S_alto*np.exp(-(x/S_alto))
    caida = S_alto*(S_alto/caida[x_inicio_caida])*np.exp(-(x/S_alto))
    
    subida = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)
    
    estimulo = np.where(tiempo_estimulo<tiempo_bajada, subida, caida)

    #Integramos
    condiciones_iniciales = [0, 0, 0]
    tiempo_min = tiempo_bajada + 20
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo, estimulo)
    params = [C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1]
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    #Devuelvo
    return np.array([tiempo, variables, tiempo_estimulo, estimulo])
