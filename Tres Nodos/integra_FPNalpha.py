#Integra el feedback positivo negativo

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#%%
def integra_FPN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, C2, dYX2, dX2, Ty2, dy2, TY2, dY2, alpha, tiempo_max):
    '''
    Las condiciones iniciales que usa son A=0 y B=0.
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
        C2 = params[8]
        dYX2 = params[9]
        dX2 = params[10]
        Ty2 = params[11]
        dy2 = params[12]
        TY2 = params[13]
        dY2 = params[14]
        alpha = params[15]

        # Variables
        X=vars[0]
        y=vars[1]
        Y=vars[2]
    
        # Sistema de ecuaciones
        dX = ((1-alpha)*C1+alpha*C2+alpha*(X**2))/(1+alpha*(X**2)) - ((1-alpha)*dYX1*KYX1+alpha*dYX2)*Y*(X/(1+((1-alpha)*X)/KYX1)) - ((1-alpha)*dX1+alpha*dX2)*X        
        dy = ((1-alpha)*Ty1+alpha*Ty2)*(X/(1+X)) - ((1-alpha)*dy1+alpha*dy2)*y        
        dY = ((1-alpha)*TY1+alpha*TY2)*y - ((1-alpha)*dy1+alpha*dY2)*Y
        
        return np.array([dX, dy, dY])

    #Integramos
    condiciones_iniciales = [0, 0, 0]
    params = [C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, C2, dYX2, dX2, Ty2, dy2, TY2, dY2, alpha]
    tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)

    return tiempo, variables

def integra_FPN_estimulo(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, C2, dYX2, dX2, Ty2, dy2, TY2, dY2, alpha, tiempo_max=1000, S_alto=100, S_bajo=0.1, N_estimulo=10000, tiempo_max_estimulo=1000, tiempo_subida=100, tiempo_bajada=300):
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
        C2 = params[8]
        dYX2 = params[9]
        dX2 = params[10]
        Ty2 = params[11]
        dy2 = params[12]
        TY2 = params[13]
        dY2 = params[14]
        alpha = params[15]

        # Variables
        X=vars[0]
        y=vars[1]
        Y=vars[2]
    
        # Sistema de ecuaciones
        dX = ((1-alpha)*C1+alpha*C2+alpha*(X**2))/(1+alpha*(X**2)) - ((1-alpha)*dYX1*KYX1+alpha*dYX2)*Y*(X/(1+((1-alpha)*X)/KYX1)) - ((1-alpha)*dX1+alpha*dX2)*X        
        dy = ((1-alpha)*Ty1+alpha*Ty2)*(X/(1+X)) - ((1-alpha)*dy1+alpha*dy2)*y        
        dY = ((1-alpha)*TY1+alpha*TY2)*y - ((1-alpha)*dy1+alpha*dY2)*Y
        
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
    params = [C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, C2, dYX2, dX2, Ty2, dy2, TY2, dY2, alpha]
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    #Devuelvo
    return np.array([tiempo, variables, tiempo_estimulo, estimulo])
