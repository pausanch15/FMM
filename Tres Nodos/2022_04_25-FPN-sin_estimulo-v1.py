#Intento integrar el modelo de Feedback Positivo Negativo de tres nodos

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPN as fpn
plt.ion()

#%%
#Integramos
#Primero con los parámetros que propone el paper
C2 = 0.015
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1

tiempo, variables = fpn.integra_FPN(C2, dYX2, dX2, Ty2, dy2, TY2, dY2, tiempo_max=100000)

X, y, Y = variables

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.grid()
plt.title('Con los Valores del Paper')
plt.legend()
plt.show()

