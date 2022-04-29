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
#Primero con los parámetros que propone el paper: con estos valores me da un desastre
C2 = 0.015
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
alpha = 0.5

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

#Intento encontrar valores para estar en el régimen oscilatorio. Pruebo cerca de estos valores, y cambiando segun lo que veo en los valores para el feedback negativo
C2 = 4.5
dYX2 = 7
dX2 = 0.01
Ty2 = 0.35
dy2 = 0.4
TY2 = 0.67
dY2 = 0.25

tiempo, variables = fpn.integra_FPN(C2, dYX2, dX2, Ty2, dy2, TY2, dY2, tiempo_max=100000)

X, y, Y = variables

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.grid()
plt.title('Cambiando los Valores')
plt.legend()
plt.show()
