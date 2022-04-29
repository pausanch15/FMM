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

#%%
def escalon(resolucion, ti, tf, S_min, S_max, tau, ts, tb):    
    #Eje del tiempo
    tiempo = np.linspace(ti, tf, resolucion)
    
    #Eje de estímulo
    S = np.ones(resolucion)
    
    #Hago el escalon
    ts = int(ts*(1/tf)*resolucion)
    tb = int(tb*(1/tf)*resolucion)
    S[0:ts] = S_min
    S[ts:tb] = S_max
    S[tb:] = S_min
    
    #Ahora la bajada exponencial
    bajada = S_max*np.exp((-tiempo)/tau)
    S[tb:] = bajada[:resolucion-tb]

    #Devuelvo el escalon
    return tiempo, S

def integra_FN_estimulo(dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max=1000, resolucion=1000, ti=0, tf=100, S_min=0, S_max=3, tau=10, ts=30, tb=50, condiciones_iniciales=[0, 0, 0]):
    '''
    Las condiciones iniciales que usa son X=y=Y=0
    Devuelve (tiempo, variables, tiempo_estimulo, estimulo)
    '''
    #Defino el modelo
    def modelo(vars, params, interpolar_estimulo, tiempo):
        #Parámetros
        C1 = interpolar_estimulo(tiempo) #aca lo interpola
        dX1 = params[0]
        dYX1 = params[1]
        KYX1 = params[2]
        Ty1 = params[3] 
        dy1 = params[4]
        TY1 = params[5]
        dY1 = params[6]

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
    tiempo_estimulo, estimulo = escalon(resolucion, ti, tf, S_min, S_max, tau, ts, tb)
    
    #Integramos
    # tiempo_min = tb + 20
    tiempo_min = tf-1
    tiempo_max = tf-1
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo, estimulo)
    params = [dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1]
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    #Devuelvo
    return np.array([tiempo, variables, tiempo_estimulo, estimulo])
