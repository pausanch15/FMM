#Integra el feedback negativo

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
plt.ion()

#%%
def integra_FN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max, pasos):
    '''
    Las condiciones iniciales que usa son A=0 y B=0.
    '''
    #Defino el modelo
    def modelo(vars, params):
        #Paràmetros
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
    condiciones_iniciales = [0, 0]
    params = [C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1]
    tiempo, variables = rk.integrar(modelo, *params, condiciones_iniciales, tiempo_max)

    return tiempo, variables
