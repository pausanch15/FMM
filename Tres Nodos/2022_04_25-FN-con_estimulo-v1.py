#Intento integrar el modelo de Feedback Negativo de tres nodos con un estímulo

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FN as fn
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#%%
#Integramos
#Primero con los parámetros que propone el paper
C1 = 4.5
dX1 = 0.01
dYX1 = 7
KYX1 = 0.04
Ty1 = 0.35
dy1 = 0.4
TY1 = 0.67
dY1 = 0.25

tiempo, variables, tiempo_estimulo, estimulo = fn.integra_FN_estimulo(dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max=1000, resolucion=1000, ti=0, tf=200, S_min=0, S_max=2, tau=20, ts=20, tb=70, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.grid()
plt.title('Con los Valores del Paper')
plt.legend()
plt.show()

