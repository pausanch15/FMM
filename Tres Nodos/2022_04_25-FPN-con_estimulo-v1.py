#Intento integrar el modelo de Feedback Positivo Negativo de tres nodos

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPN as fpn
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#%%
#Integramos
#Primero con los parámetros que propone el paper
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1

tiempo, variables, tiempo_estimulo, estimulo = fpn.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables

#Para tratar de analizar la caida, disminuyo la amplitud de las oscilaciones en un factor epsilon
epsilon = 0.01
X, y, Y = variables*epsilon
 
plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.grid()
plt.title('FPN Con los Valores del Paper')
plt.legend()
plt.show()
